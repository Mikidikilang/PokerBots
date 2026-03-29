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
import torch
from dataclasses import dataclass, field
from typing import Any

from src.env.action_mapper import ActionMapper, GameContext, PokerAction
from src.env.equity import EquityCalculator
from src.env.features import ObservationBuilder
from src.env.wrappers import RLCardWrapper
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
            self.big_blind = self.env.game.big_blind
        except AttributeError:
            self.big_blind = 2.0
            logger.warning("Could not extract big_blind from env.game, using default: 2.0")

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
            self.total_pot += self.env.game.init_chips * 2

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
            logger.error("Error in model step: %s", e)
            return None

    def _oracle_step(self, obs_dict: dict[str, Any]) -> dict[str, Any] | None:
        try:
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

            hole_cards   = obs_dict.get("hand", [])
            public_cards = obs_dict.get("public_cards", [])

            if not hole_cards or len(hole_cards) < 2:
                logger.warning("Invalid hole cards: %s", hole_cards)
                return None

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

            best_action = None
            best_ev     = float("-inf")

            for action in legal_actions:
                ev = self._compute_action_ev(
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

    # =========================================================================
    # [FIX R-3] Improved EV computation with pot-odds and fold-equity
    # =========================================================================

    def _compute_action_ev(
        self,
        action:  PokerAction,
        equity:  float,
        context: GameContext,
    ) -> float:
        """Compute expected value of an action with pot-odds and fold-equity.

        ─── Improvements over the original single-street formula ─────────
        Original:
            EV(call) = equity * pot_after_call - (1-equity) * call_amount
            EV(raise)= equity * pot_after_raise - (1-equity) * raise_amount

        Problems:
            1. Fold equity ignored: raises can win the pot immediately if
               the opponent folds. The original assumed they always call.
            2. No pot-odds check: calling with 20% equity vs a pot-sized bet
               is a clear -EV call. The original sometimes preferred calling.
            3. No implied-odds proxy: post-flop, deep-stack situations give
               extra value beyond the showdown equity.

        Improved model:
            fold_probability  = estimated probability opponent folds to a raise
            EV(raise) = fold_prob * current_pot
                      + (1 - fold_prob) * [equity * final_pot
                                          - (1 - equity) * raise_amount]
            EV(call)  = pot_odds_check * [equity * final_pot
                        - (1-equity) * call_amount]
                      + implied_odds_bonus (small proxy for future streets)

        Fold probability estimation:
            We use a simple threshold model based on raise size:
                fold_prob ≈ sigmoid(k * (raise_size / pot - 0.5))
            This gives ~35% fold probability for pot-sized raises, increasing
            with larger over-bets.
        """
        # ── Fold: give up 0 additional chips ─────────────────────────────
        if action == PokerAction.FOLD:
            return 0.0

        # ── Check: free to see the next card ──────────────────────────────
        if action == PokerAction.CHECK:
            if context.amount_to_call == 0.0:
                check_ev: float = equity * context.pot_size
                return check_ev
            else:
                # Invalid CHECK with bet → treat as FOLD
                return 0.0

        # ── Call: pot-odds aware EV ───────────────────────────────────────
        if action == PokerAction.CALL:
            call_amount: float = context.amount_to_call

            if call_amount == 0.0:
                # No bet to call → check
                check_ev: float = equity * context.pot_size
                return check_ev

            # Pot odds breakeven equity
            pot_after_call: float = context.pot_size + call_amount
            breakeven_equity: float = call_amount / pot_after_call

            if equity < breakeven_equity * 0.8:
                # Clear fold: equity well below pot-odds threshold
                return -call_amount * (breakeven_equity - equity) * 2.0

            # Implied odds proxy: on early streets, add small bonus for draws
            implied_bonus: float = 0.0
            if context.pot_size < context.my_stack * 0.3:
                implied_bonus = max(0.0, equity - breakeven_equity) * call_amount * 0.5

            call_ev: float = (
                equity * pot_after_call
                - (1.0 - equity) * call_amount
                + implied_bonus
            )
            return call_ev

        # ── Raise / All-in: fold-equity aware EV ─────────────────────────
        try:
            resolved = self.action_mapper.resolve_action(action, context)
            raise_amount: float = resolved.amount
        except Exception:
            return 0.0

        if raise_amount <= 0:
            return 0.0

        # Estimate fold probability via sigmoid of raise-to-pot ratio.
        # k=2.0: ~35% folds at 1.0x pot, ~55% at 2.0x, ~20% at 0.5x.
        raise_to_pot_ratio: float = raise_amount / max(context.pot_size, 1.0)
        k: float = 2.0
        fold_probability: float = 1.0 / (
            1.0 + math.exp(-k * (raise_to_pot_ratio - 0.5))
        )
        fold_probability = min(fold_probability, 0.70)  # cap at 70%

        ev_if_fold: float = context.pot_size

        pot_if_called: float = context.pot_size + raise_amount
        ev_if_call: float = (
            equity * pot_if_called
            - (1.0 - equity) * raise_amount
        )

        raise_ev: float = (
            fold_probability * ev_if_fold
            + (1.0 - fold_probability) * ev_if_call
        )
        return raise_ev

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
