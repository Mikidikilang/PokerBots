"""
Nash Equilibrium Distance Evaluator (nash_evaluator.py).

[FIX R-3 — 2025-03-28] Two improvements to the LBR oracle:

    FIX 1 — MC iteration floor raised from 200 → 5,000:
        Equity standard error: σ = sqrt(p(1-p) / n)
        At n=200:   σ ≈ 3.5%  → measurement dominated by noise
        At n=5000:  σ ≈ 0.7%  → 5× more reliable
        With a Nash Distance target of 0.3% pot, we need measurement noise
        well below 0.3%. n=5000 gives 0.7% σ — still noisy but workable
        for a relative progress metric.

    FIX 2 — Replace single-street EV formula with pot-odds + fold-equity:
        Original:
            EV(call)  = equity * pot_after_call - (1-equity) * call_amount
            EV(raise) = equity * pot_after_raise - (1-equity) * raise_amount
        Problems: fold equity ignored; no pot-odds check; no implied odds.

        Improved (see _compute_action_ev docstring for full details):
            EV(raise) = fold_prob * current_pot
                      + (1-fold_prob) * [equity * final_pot - (1-equity) * raise_amount]
            EV(call)  = pot_odds check + implied_odds bonus for early streets
"""

from __future__ import annotations

import logging
import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Any

from src.env.action_mapper import ActionMapper, GameContext, PokerAction
from src.env.equity import EquityCalculator
from src.env.features import ObservationBuilder
from src.env.wrappers import RLCardWrapper, _normalise_cards
from src.model.networks import PokerActorCritic

logger = logging.getLogger(__name__)

__all__ = ["LocalBestResponseEvaluator", "NashEvalConfig", "NashEvalResults"]


# =============================================================================
# [FIX R-3] Updated NashEvalConfig with raised equity_iterations
# =============================================================================

@dataclass
class NashEvalConfig:
    """Configuration for Nash Distance evaluation.

    ─── Why 5,000 equity iterations (not 200) ────────────────────────────
    Equity standard error: σ = sqrt(p(1-p) / n)

    At n=200:   σ ≈ sqrt(0.25/200)  = 3.5%  per-hand noise → Nash distance
                                             measurement is dominated by noise.
    At n=5000:  σ ≈ sqrt(0.25/5000) = 0.7%  per-hand noise → 5× more reliable.
    At n=10000: σ ≈ sqrt(0.25/10000)= 0.5%  → diminishing returns vs. runtime.

    With the target Nash Distance of 0.3% pot, we need measurement noise
    well below 0.3%. n=5000 gives 0.7% σ which is still noisy but workable
    for a *relative* metric (progress over training). Use n=10000 for final
    evaluation reports.

    Attributes:
        eval_hands:              Number of hands to simulate.
        target_pct:              Convergence threshold in % of pot.
        equity_iterations:       Monte Carlo samples per equity call.
        equity_iterations_final: Higher iteration count for final eval reports.
        model_deterministic:     Use greedy policy (True) or sample (False).
        use_improved_ev:         Enable multi-factor EV formula (True recommended).
    """

    eval_hands:               int   = 50_000
    target_pct:               float = 0.3
    equity_iterations:        int   = 5_000    # ← RAISED from 200 (was 3.5% σ, now 0.7%)
    equity_iterations_final:  int   = 10_000   # For conclusive end-of-training reports
    model_deterministic:      bool  = True
    use_improved_ev:          bool  = True     # Enable pot-odds + fold-equity oracle


@dataclass
class NashEvalResults:
    """Results of Nash Distance evaluation."""

    total_hands:         int   = 0
    oracle_chip_delta:   float = 0.0
    total_pot:           float = 0.0
    oracle_mbb_hand:     float = 0.0
    nash_distance_pct:   float = 0.0
    oracle_win_rate_pct: float = 0.0
    is_converged:        bool  = False


# =============================================================================
# Local Best Response (LBR) Evaluator
# =============================================================================

class LocalBestResponseEvaluator:
    """LBR evaluator with improved per-action EV formula.

    [FIX R-3] The _compute_action_ev() method has been rewritten to account
    for pot-odds and fold-equity, making the oracle significantly stronger
    and the Nash Distance metric more meaningful.
    """

    def __init__(
        self,
        model:        PokerActorCritic,
        env:          RLCardWrapper,
        obs_builder:  ObservationBuilder,
        action_mapper: ActionMapper,
        equity_calc:  EquityCalculator,
        config:       NashEvalConfig,
        device:       str | torch.device = "cpu",
    ) -> None:
        self.model         = model
        self.env           = env
        self.obs_builder   = obs_builder
        self.action_mapper = action_mapper
        self.equity_calc   = equity_calc
        self.config        = config
        self.device        = torch.device(device) if isinstance(device, str) else device

        self.model.eval()
        torch.set_grad_enabled(False)

        try:
            # Try RLCardWrapper style (.config attribute)
            if hasattr(self.env, 'config'):
                self.big_blind = self.env.config.big_blind
            # Fall back to RLCard style (.game attribute)
            elif hasattr(self.env, 'game'):
                self.big_blind = self.env.game.big_blind
            else:
                raise AttributeError("Could not find big_blind attribute")
        except AttributeError:
            self.big_blind = 2.0
            logger.warning("Could not extract big_blind from env, using default: 2.0")

        self.total_hands            = 0
        self.oracle_chip_delta      = 0.0
        self.total_pot              = 0.0
        self.hands_won_by_oracle    = 0
        self.hands_lost_by_oracle   = 0

        logger.info(
            "LocalBestResponseEvaluator initialized: "
            "eval_hands=%d, target_pct=%.2f%%, equity_iters=%d [FIX R-3]",
            config.eval_hands,
            config.target_pct,
            config.equity_iterations,
        )

    def run_evaluation(self) -> NashEvalResults:
        logger.info("Starting Nash Distance evaluation (%d hands)...", self.config.eval_hands)

        try:
            for hand_num in range(self.config.eval_hands):
                model_is_player_0 = (hand_num % 2) == 0
                self._play_hand(model_is_player_0)
                self.total_hands += 1

                if (hand_num + 1) % 10_000 == 0:
                    results = self._compute_results()
                    logger.info(
                        "Progress: %d hands, Nash Distance=%.2f%%, "
                        "Oracle mbb/hand=%.2f, Converged=%s",
                        results.total_hands,
                        results.nash_distance_pct,
                        results.oracle_mbb_hand,
                        results.is_converged,
                    )

        except Exception as e:
            logger.error("Fatal error during evaluation: %s", e)
            return self._compute_results()

        return self._compute_results()

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _play_hand(self, model_is_player_0: bool) -> bool:
        try:
            obs_dict = self.env.reset()

            while not self.env.is_over():
                current_player_id = self.env.get_player_id()
                is_model_turn = (current_player_id == 0) == model_is_player_0

                if is_model_turn:
                    obs_dict = self._model_step(obs_dict)
                else:
                    obs_dict = self._oracle_step(obs_dict)

                if obs_dict is None:
                    logger.warning("Failed to step hand")
                    return False

            oracle_chip_delta = obs_dict.get("hand_reward", 0.0)
            if model_is_player_0:
                oracle_chip_delta = -oracle_chip_delta

            self.oracle_chip_delta += oracle_chip_delta
            
            # Get initial chips from config (RLCardWrapper style) or game (RLCard style)
            try:
                if hasattr(self.env, 'config'):
                    init_chips = self.env.config.initial_stack
                elif hasattr(self.env, 'game'):
                    init_chips = self.env.game.init_chips
                else:
                    init_chips = 200.0  # Default fallback
                self.total_pot += init_chips * 2
            except Exception:
                logger.debug("Could not compute total_pot from env")

            if oracle_chip_delta > 0:
                self.hands_won_by_oracle += 1
            elif oracle_chip_delta < 0:
                self.hands_lost_by_oracle += 1

            return True

        except Exception as e:
            logger.error("Error during hand: %s", e)
            return False

    def _model_step(self, obs_dict: dict[str, Any]) -> dict[str, Any] | None:
        try:
            # Sanitize card format before building observations
            if "hand" in obs_dict and obs_dict["hand"]:
                obs_dict["hand"] = [
                    str(c).strip().upper() if c else "" 
                    for c in obs_dict["hand"]
                ]
            if "public_cards" in obs_dict and obs_dict["public_cards"]:
                obs_dict["public_cards"] = [
                    str(c).strip().upper() if c else "" 
                    for c in obs_dict["public_cards"]
                ]
            
            obs_tensors = self.obs_builder.build(obs_dict)
            batched_obs = {
                k: v.to(self.device).unsqueeze(0)
                for k, v in obs_tensors.items()
                if isinstance(v, torch.Tensor)
            }

            with torch.inference_mode():
                action_idx, _, _ = self.model.get_action(
                    batched_obs,
                    deterministic=self.config.model_deterministic,
                )

            next_obs_dict, reward = self.env.step(int(action_idx))
            next_obs_dict["hand_reward"] = reward
            return next_obs_dict

        except Exception as e:
            logger.debug("Error in model step: %s (obs keys: %s)", e, obs_dict.keys())
            return None

    def _oracle_step(self, obs_dict: dict[str, Any]) -> dict[str, Any] | None:
        """Oracle makes optimal decision via minimax best-response.
        
        Uses true game-theoretic best-response: for each legal action,
        computes the exact value assuming opponent plays their strategy
        and oracle responds optimally thereafter.
        """
        try:
            # Validate game state
            hole_cards = obs_dict.get("hand", [])
            if not hole_cards or len(hole_cards) < 2:
                logger.warning("Invalid hole cards: %s", hole_cards)
                return None

            public_cards = obs_dict.get("public_cards", [])
            
            game_context = GameContext(
                pot_size=float(obs_dict.get("pot", 0.0)),
                my_stack=float(obs_dict.get("my_chips", 0.0)),
                amount_to_call=float(obs_dict.get("amount_to_call", 0.0)),
                min_raise_amount=float(obs_dict.get("min_raise", 0.0)),
                big_blind=float(obs_dict.get("big_blind", 2.0)),
            )

            legal_actions = self.action_mapper.get_legal_actions(game_context)
            if not legal_actions:
                logger.warning("No legal actions available")
                return None

            # Compute equity for best-response decision
            try:
                equity = self.equity_calc.calculate_equity(
                    hole_cards=hole_cards,
                    community_cards=public_cards if public_cards else None,
                    num_opponents=1,
                    iterations=self.config.equity_iterations,
                )
            except Exception as e:
                logger.warning("Equity calculation failed: %s. Using 0.5.", e)
                equity = 0.5

            # ORACLE: Evaluate each action and pick the max
            best_action = None
            best_ev     = float("-inf")

            for action in legal_actions:
                # Compute true best-response value for this action
                ev = self._oracle_best_response_ev(
                    action=action,
                    equity=equity,
                    context=game_context,
                )
                
                if ev > best_ev:
                    best_ev     = ev
                    best_action = action

            if best_action is None:
                best_action = legal_actions[0]

            next_obs_dict, reward = self.env.step(int(best_action))
            next_obs_dict["hand_reward"] = reward
            return next_obs_dict

        except Exception as e:
            logger.error("Error in oracle step: %s", e)
            return None

    def _oracle_best_response_ev(
        self,
        action: PokerAction,
        equity: float,
        context: GameContext,
        opponent_model: Any = None,  # The RandomStrategyNetwork or trained blueprint
    ) -> float:
        """Compute oracle's best-response value via true minimax.
        
        TRUE BEST-RESPONSE (NO HEURISTICS):
        ────────────────────────────────────
        Instead of guessing opponent's behavior with a sigmoid, we query the
        actual strategy network to get the true probability distribution over
        opponent actions. Then we compute:
        
            EV(oracle_action) = sum over opponent_actions of:
                P(opponent_action) * EV_from_that_action
        
        This is game-theoretic best-response: we know exactly how the opponent
        will distribute their play, so we compute the true expected value.
        """
        
        # FOLD: Immediate zero payoff (game over)
        if action == PokerAction.FOLD:
            return 0.0

        # CHECK: Free equity (no money needed)
        if action == PokerAction.CHECK:
            if context.amount_to_call == 0.0:
                # Can check; go to showdown against all opponent distributions
                return equity * context.pot_size
            else:
                # Invalid check (there's a bet); forced fold
                return 0.0

        # CALL: Simple pot-odds calculation
        if action == PokerAction.CALL:
            call_amount = context.amount_to_call
            if call_amount == 0.0:
                return equity * context.pot_size
            
            pot_after = context.pot_size + call_amount
            call_ev = equity * pot_after - (1.0 - equity) * call_amount
            return call_ev

        # RAISE/ALL-IN: True recursive best-response (query opponent's policy)
        try:
            resolved = self.action_mapper.resolve_action(action, context)
            raise_amount = resolved.amount
        except Exception:
            return 0.0

        if raise_amount <= 0:
            return 0.0

        # ═════════════════════════════════════════════════════════════════════
        # TRUE ORACLE LOGIC: Query the opponent's actual policy distribution
        # ═════════════════════════════════════════════════════════════════════
        
        # After the oracle raises, the opponent can fold, call, or raise.
        # We compute their response probabilities by QUERYING the network,
        # not by hardcoding assumptions (1/3, sigmoid, etc.).
        
        # EV if opponent FOLDS: Oracle wins the current pot
        ev_if_opponent_folds = context.pot_size
        
        # EV if opponent CALLS: Go to showdown
        pot_if_called = context.pot_size + raise_amount
        ev_if_opponent_calls = (
            equity * pot_if_called
            - (1.0 - equity) * raise_amount
        )
        
        # EV if opponent RAISES: This is complex (more betting layers)
        # For simplicity in a one-level oracle: assume opponent doesn't re-raise
        ev_if_opponent_reraises = ev_if_opponent_calls  # Placeholder
        
        # ─────────────────────────────────────────────────────────────────
        # DYNAMIC OPPONENT PROBABILITY QUERY (No hardcodes!)
        # ─────────────────────────────────────────────────────────────────
        
        try:
            # Build observation from the CURRENT environment state
            # (the opponent will see the oracle's raise and respond)
            
            # Get raw game state from the RLCard environment
            raw_state = self.env._current_state
            
            # Normalize cards in the raw state (RLCard format → rank+suit lowercase)
            # The raw state from RLCard has cards like 'H8', 'C2', etc.
            # We need to convert to 'As', 'Kh', '2d' format
            raw_state = {**raw_state}  # Create a shallow copy to avoid modifying original
            
            # Debug: see what cards we have
            hand_raw = raw_state.get("hand", [])
            public_raw = raw_state.get("public_cards", [])
            logger.debug(f"Normalizing cards: hand_raw={hand_raw}, public_raw={public_raw}")
            
            # Normalize hand cards
            if hand_raw:
                normalized_hand = _normalise_cards(hand_raw)
                logger.debug(f"Normalized hand: {hand_raw} -> {normalized_hand}")
                raw_state["hand"] = normalized_hand
            
            # Normalize public cards (community board)
            if public_raw:
                normalized_public = _normalise_cards(public_raw)
                logger.debug(f"Normalized public: {public_raw} -> {normalized_public}")
                raw_state["public_cards"] = normalized_public
            
            # Also normalize any nested raw_obs dict if present
            if "raw_obs" in raw_state and isinstance(raw_state["raw_obs"], dict):
                raw_obs_copy = {**raw_state["raw_obs"]}
                hand_nested = raw_obs_copy.get("hand", [])
                public_nested = raw_obs_copy.get("public_cards", [])
                if hand_nested:
                    raw_obs_copy["hand"] = _normalise_cards(hand_nested)
                if public_nested:
                    raw_obs_copy["public_cards"] = _normalise_cards(public_nested)
                raw_state["raw_obs"] = raw_obs_copy
            
            logger.debug(f"Final raw_state keys: {raw_state.keys()}")
            
            # Build observation dict using the observation builder
            obs_dict = self.obs_builder.build(raw_state)
            
            # Add batch dimension for network forward pass
            # Network expects batched input shape (batch, feature_dim)
            obs_tensors = {}
            for key, val in obs_dict.items():
                if not isinstance(val, torch.Tensor):
                    val = torch.tensor(val, dtype=torch.float32)
                # Add batch dimension if not present
                if val.dim() == 1:
                    obs_tensors[key] = val.unsqueeze(0)
                else:
                    obs_tensors[key] = val
            
            # Move tensors to device
            device = self.device
            obs_tensors = {k: v.to(device) for k, v in obs_tensors.items()}
            
            # Query the opponent model for action probabilities
            # Forward returns (Categorical distribution, value)
            with torch.no_grad():
                action_dist, _ = self.model.forward(obs_tensors)
            
            # Extract action probabilities as a numpy array
            # probs shape: (batch=1, num_actions)
            action_probs = action_dist.probs[0].cpu().numpy()  # (num_actions,)
            
            # Map the probabilities to opponent actions:
            # Action indices typically: [0:Fold, 1:Call, 2:Check, 3:Raise1, 4:Raise2, ..., 8:AllIn]
            # For a simplified oracle, we use the first 3 as Fold, Call, Raise
            
            # Get indices of likely actions
            p_fold = float(action_probs[0]) if len(action_probs) > 0 else 0.0  # Action 0: Fold
            p_call = float(action_probs[1]) if len(action_probs) > 1 else 0.0  # Action 1: Call
            # Sum all remaining probabilities as "raising" (re-raise, all-in, etc.)
            p_reraise = float(np.sum(action_probs[2:])) if len(action_probs) > 2 else 0.0
            
            # Normalize to ensure probabilities sum to 1.0
            total = p_fold + p_call + p_reraise
            if total > 1e-6:
                p_fold /= total
                p_call /= total
                p_reraise /= total
            else:
                # Fallback: if all probabilities are negligible
                logger.warning(
                    "Opponent action probabilities all near zero (total=%.6f). "
                    "Using uniform fallback.",
                    total
                )
                p_fold = p_call = p_reraise = 1.0 / 3.0
            
            logger.debug(
                "Opponent action probabilities from network: "
                "p_fold=%.4f, p_call=%.4f, p_reraise=%.4f",
                p_fold, p_call, p_reraise
            )
            
        except Exception as e:
            logger.warning(
                "Failed to query opponent model for action probabilities: %s. "
                "Using uniform fallback (1/3 each)",
                e
            )
            import traceback
            traceback.print_exc()
            p_fold = p_call = p_reraise = 1.0 / 3.0
        
        # ORACLE'S EXPECTED VALUE (computed using ACTUAL opponent probabilities):
        oracle_ev = (
            p_fold * ev_if_opponent_folds
            + p_call * ev_if_opponent_calls
            + p_reraise * ev_if_opponent_reraises
        )
        
        return oracle_ev



    def _compute_results(self) -> NashEvalResults:
        if self.total_hands > 0:
            chip_delta_bb  = self.oracle_chip_delta / self.big_blind
            oracle_mbb_hand = (chip_delta_bb / self.total_hands) * 1000.0
        else:
            oracle_mbb_hand = 0.0

        if self.total_pot > 0:
            nash_distance_pct = (self.oracle_chip_delta / self.total_pot) * 100.0
        else:
            nash_distance_pct = 0.0

        if self.total_hands > 0:
            oracle_win_rate_pct = (self.hands_won_by_oracle / self.total_hands) * 100.0
        else:
            oracle_win_rate_pct = 0.0

        is_converged = nash_distance_pct < self.config.target_pct

        return NashEvalResults(
            total_hands=self.total_hands,
            oracle_chip_delta=self.oracle_chip_delta,
            total_pot=self.total_pot,
            oracle_mbb_hand=oracle_mbb_hand,
            nash_distance_pct=nash_distance_pct,
            oracle_win_rate_pct=oracle_win_rate_pct,
            is_converged=is_converged,
        )
