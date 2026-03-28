"""
Nash Equilibrium Distance Evaluator (nash_evaluator.py).

Implements an approximate exploitability metric using a heuristic Local Best Response
(LBR) Oracle. The LBR oracle leverages the EquityCalculator to compute hand strength
and then selects EV-maximizing actions against the model player.

This evaluator is critical for Phase 2 (Co-Adaptive FSP) convergence detection:
    - Nash Distance % measures how exploitable the model is (lower is better).
    - Phase 2 terminates when Nash Distance < target_pct (default 0.3%).

Architecture:
    - Model plays against LBR Oracle in self-play environment (RLCardWrapper).
    - Alternates positions each hand to ensure positional fairness.
    - Model uses neural network inference (deterministic greedy action).
    - Oracle uses equity calculations + EV heuristic for action selection.
    - Tracks cumulative chip delta from Oracle's perspective.
    - Calculates exploitability metrics: mbb/hand and Nash Distance %.

Performance Optimizations:
    - EquityCalculator uses low iteration count (default 200) to keep eval fast.
    - Batched network inference with torch.inference_mode() (no gradient).
    - Minimal state copying; observation dicts used in-place.
"""

from __future__ import annotations

import logging
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
# Configuration and Results Data Structures
# =============================================================================

@dataclass
class NashEvalConfig:
    """Configuration for Nash Distance evaluation.

    Attributes:
        eval_hands: Number of hands to simulate (default 50,000).
        target_pct: Target Nash Distance percentage (default 0.3%).
        equity_iterations: Monte Carlo iterations for equity calculation (lower=faster).
        model_deterministic: Use greedy policy (True) or sample (False).
    """

    eval_hands: int = 50_000
    target_pct: float = 0.3
    equity_iterations: int = 200
    model_deterministic: bool = True


@dataclass
class NashEvalResults:
    """Results of Nash Distance evaluation.

    Attributes:
        total_hands: Total hands played.
        oracle_chip_delta: Cumulative chips won by oracle (from oracle's perspective).
        total_pot: Total chips in all pots (for Nash Distance % calculation).
        oracle_mbb_hand: Oracle win rate in milli-big-blinds per hand.
        nash_distance_pct: Exploitability metric (Oracle chips / Total pot) * 100.
        oracle_win_rate_pct: Oracle win rate as percentage (hands_won / total_hands) * 100.
        is_converged: True if nash_distance_pct < target_pct.
    """

    total_hands: int = 0
    oracle_chip_delta: float = 0.0
    total_pot: float = 0.0
    oracle_mbb_hand: float = 0.0
    nash_distance_pct: float = 0.0
    oracle_win_rate_pct: float = 0.0
    is_converged: bool = False


# =============================================================================
# Local Best Response (LBR) Evaluator
# =============================================================================

class LocalBestResponseEvaluator:
    """Evaluates model exploitability using a heuristic LBR oracle.

    The LBR oracle uses hand equity (via EquityCalculator) to make EV-optimal
    decisions. By playing many hands against the oracle and tracking its chip
    delta, we approximate the model's exploitability relative to the oracle's
    strategy.

    The metric "Nash Distance %" represents the percentage of the total pot
    that the oracle wins, which correlates with how far the model is from
    Nash equilibrium.
    """

    def __init__(
        self,
        model: PokerActorCritic,
        env: RLCardWrapper,
        obs_builder: ObservationBuilder,
        action_mapper: ActionMapper,
        equity_calc: EquityCalculator,
        config: NashEvalConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the Nash evaluator.

        Args:
            model: PokerActorCritic network (must be in eval mode).
            env: RLCardWrapper self-play environment.
            obs_builder: ObservationBuilder for state-to-tensor conversion.
            action_mapper: ActionMapper for action resolution.
            equity_calc: EquityCalculator for hand strength computation.
            config: NashEvalConfig with evaluation settings.
            device: PyTorch device (cpu/cuda).
        """
        self.model = model
        self.env = env
        self.obs_builder = obs_builder
        self.action_mapper = action_mapper
        self.equity_calc = equity_calc
        self.config = config
        self.device = torch.device(device) if isinstance(device, str) else device

        # Ensure model is in eval mode and no gradients
        self.model.eval()
        torch.set_grad_enabled(False)

        # Cache big blind for later use in metrics calculation
        try:
            self.big_blind = self.env.game.big_blind
        except AttributeError:
            self.big_blind = 2.0
            logger.warning("Could not extract big_blind from env.game, using default: 2.0")

        # Tracking state
        self.total_hands = 0
        self.oracle_chip_delta = 0.0
        self.total_pot = 0.0
        self.hands_won_by_oracle = 0
        self.hands_lost_by_oracle = 0

        logger.info(
            "LocalBestResponseEvaluator initialized: "
            "eval_hands=%d, target_pct=%.2f%%, equity_iters=%d",
            config.eval_hands,
            config.target_pct,
            config.equity_iterations,
        )

    def run_evaluation(self) -> NashEvalResults:
        """Run the full Nash distance evaluation.

        Returns:
            NashEvalResults with exploitability metrics.
        """
        logger.info("Starting Nash Distance evaluation (%d hands)...", self.config.eval_hands)

        try:
            for hand_num in range(self.config.eval_hands):
                # Alternate which player is the model vs oracle each hand
                # This ensures positional fairness
                model_is_player_0 = (hand_num % 2) == 0

                hand_success = self._play_hand(model_is_player_0)
                self.total_hands += 1

                # Log progress every 10k hands
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
        """Play a single hand between model and oracle.

        Args:
            model_is_player_0: If True, model is player 0; oracle is player 1.
                              If False, model is player 1; oracle is player 0.

        Returns:
            True if hand completed successfully, False otherwise.
        """
        try:
            # Reset environment
            obs_dict = self.env.reset()

            # Play hand until terminal
            while not self.env.is_over():
                current_player_id = self.env.get_player_id()
                is_model_turn = (current_player_id == 0) == model_is_player_0

                if is_model_turn:
                    # Model's turn: use neural network
                    obs_dict = self._model_step(obs_dict)
                else:
                    # Oracle's turn: use equity-based heuristic
                    obs_dict = self._oracle_step(obs_dict)

                if obs_dict is None:
                    logger.warning("Failed to step hand")
                    return False

            # Hand is over: extract chip delta from oracle's perspective
            # env.step() returns reward from player 0's perspective
            # If oracle was player 0 (model_is_player_0 == False), use reward directly
            # If oracle was player 1 (model_is_player_0 == True), multiply by -1
            oracle_chip_delta = obs_dict.get("hand_reward", 0.0)
            if model_is_player_0:
                # Model was player 0, oracle was player 1
                # Flip the sign: env returns P0's perspective, we want P1's
                oracle_chip_delta = -oracle_chip_delta

            self.oracle_chip_delta += oracle_chip_delta
            self.total_pot += self.env.game.init_chips * 2  # Total pot in self-play

            # Track oracle wins/losses
            if oracle_chip_delta > 0:
                self.hands_won_by_oracle += 1
            elif oracle_chip_delta < 0:
                self.hands_lost_by_oracle += 1

            return True

        except Exception as e:
            logger.error("Error during hand: %s", e)
            return False

    def _model_step(self, obs_dict: dict[str, Any]) -> dict[str, Any] | None:
        """Execute one action by the model player.

        Args:
            obs_dict: Current observation dict from environment.

        Returns:
            Updated observation dict after step, or None on error.
        """
        try:
            # Convert raw obs to tensors via ObservationBuilder
            obs_tensors = self.obs_builder.build(obs_dict)

            # Prepare tensors: move to device and add batch dimension
            # Model expects batched input [batch_size, features...]
            batched_obs = {}
            for key, tensor in obs_tensors.items():
                if isinstance(tensor, torch.Tensor):
                    # Move to device and add batch dimension for model inference
                    batched_obs[key] = tensor.to(self.device).unsqueeze(0)
                else:
                    # Non-tensor values pass through unchanged
                    batched_obs[key] = tensor

            # Get model action (deterministic greedy)
            with torch.inference_mode():
                action_idx, _, _ = self.model.get_action(
                    batched_obs,
                    deterministic=self.config.model_deterministic,
                )

            # Step environment with action
            next_obs_dict, reward = self.env.step(int(action_idx))
            next_obs_dict["hand_reward"] = reward
            return next_obs_dict

        except Exception as e:
            logger.error("Error in model step: %s", e)
            return None

    def _oracle_step(self, obs_dict: dict[str, Any]) -> dict[str, Any] | None:
        """Execute one action by the LBR oracle.

        The oracle:
        1. Extracts hand and board from obs_dict
        2. Calculates equity using EquityCalculator
        3. Computes EV for each legal action
        4. Selects action with maximum EV
        5. Steps environment

        Args:
            obs_dict: Current observation dict from environment.

        Returns:
            Updated observation dict after step, or None on error.
        """
        try:
            # Extract game context from observation
            game_context = GameContext(
                pot_size=float(obs_dict.get("pot", 0.0)),
                my_stack=float(obs_dict.get("my_chips", 0.0)),
                amount_to_call=float(obs_dict.get("amount_to_call", 0.0)),
                min_raise_amount=float(obs_dict.get("min_raise", 0.0)),
                big_blind=float(obs_dict.get("big_blind", 2.0)),
            )

            # Get legal actions
            legal_actions = self.action_mapper.get_legal_actions(game_context)
            if not legal_actions:
                logger.warning("No legal actions available")
                return None

            # Extract oracle's hand and board
            hole_cards = obs_dict.get("hand", [])
            public_cards = obs_dict.get("public_cards", [])

            if not hole_cards or len(hole_cards) < 2:
                logger.warning("Invalid hole cards: %s", hole_cards)
                return None

            # Calculate equity for the oracle
            # Assume 1 opponent (HUNL or heads-up situation)
            try:
                equity = self.equity_calc.calculate_equity(
                    hole_cards=hole_cards,
                    community_cards=public_cards if public_cards else None,
                    num_opponents=1,
                    iterations=self.config.equity_iterations,
                )
            except Exception as e:
                logger.warning(
                    "Equity calculation failed for %s vs board %s: %s. Using 0.5 fallback.",
                    hole_cards, public_cards if public_cards else "empty", e
                )
                equity = 0.5

            # Compute EV for each legal action
            best_action = None
            best_ev = float("-inf")

            for action in legal_actions:
                ev = self._compute_action_ev(
                    action=action,
                    equity=equity,
                    context=game_context,
                )

                if ev > best_ev:
                    best_ev = ev
                    best_action = action

            if best_action is None:
                # Fallback: choose first legal action
                best_action = legal_actions[0]
                logger.warning("No best action found, fallback to: %s", best_action)

            # Step environment with oracle's action
            next_obs_dict, reward = self.env.step(int(best_action))
            next_obs_dict["hand_reward"] = reward
            return next_obs_dict

        except Exception as e:
            logger.error("Error in oracle step: %s", e)
            return None

    def _compute_action_ev(
        self,
        action: PokerAction,
        equity: float,
        context: GameContext,
    ) -> float:
        """Compute expected value of an action given hand equity.

        Simple heuristic EV calculation:
        - Fold: EV = 0 (lose any posted amount)
        - Check/Call: EV = equity * (pot_after_call) - (1 - equity) * call_amount
        - Raise/All-in: EV similar, assuming opponent calls with full range

        Args:
            action: PokerAction to evaluate.
            equity: Hand strength [0.0, 1.0] from equity calculator.
            context: GameContext for calculating amounts.

        Returns:
            Estimated EV of the action.
        """
        if action == PokerAction.FOLD:
            # Folding loses 0 additional chips (but loses any posted amount)
            # EV is 0.0 (already committed; no future recovery)
            return 0.0

        elif action == PokerAction.CHECK_CALL:
            # If we call, we put in additional chips, and pot increases
            call_amount = context.amount_to_call
            pot_after_call = context.pot_size + call_amount

            # EV = prob(win) * pot_after_call - prob(lose) * call_amount
            # This assumes showdown; intermediate position results assumed 0 for simplicity
            ev = equity * pot_after_call - (1.0 - equity) * call_amount
            return ev

        else:
            # Raise / All-in actions
            # Resolve the action to get actual chip amount
            try:
                resolved = self.action_mapper.resolve_action(action, context)
                raise_amount = resolved.amount

                # EV for raise: assume opponent calls (simplified heuristic)
                # EV = equity * (pot + raise) - (1 - equity) * raise
                pot_after_raise = context.pot_size + raise_amount
                ev = equity * pot_after_raise - (1.0 - equity) * raise_amount
                return ev

            except Exception:
                # If resolution fails, fallback to 0
                logger.debug("Failed to resolve action %s", action)
                return 0.0

    def _compute_results(self) -> NashEvalResults:
        """Compute final Nash Distance metrics from accumulated state.

        Returns:
            NashEvalResults with all exploitability metrics.
        """
        # Compute mbb/hand
        if self.total_hands > 0:
            chip_delta_bb = self.oracle_chip_delta / self.big_blind
            oracle_mbb_hand = (chip_delta_bb / self.total_hands) * 1000.0
        else:
            oracle_mbb_hand = 0.0

        # Compute Nash Distance %
        if self.total_pot > 0:
            nash_distance_pct = (self.oracle_chip_delta / self.total_pot) * 100.0
        else:
            nash_distance_pct = 0.0

        # Compute win rate %
        if self.total_hands > 0:
            oracle_win_rate_pct = (self.hands_won_by_oracle / self.total_hands) * 100.0
        else:
            oracle_win_rate_pct = 0.0

        # Check convergence
        is_converged = nash_distance_pct < self.config.target_pct

        results = NashEvalResults(
            total_hands=self.total_hands,
            oracle_chip_delta=self.oracle_chip_delta,
            total_pot=self.total_pot,
            oracle_mbb_hand=oracle_mbb_hand,
            nash_distance_pct=nash_distance_pct,
            oracle_win_rate_pct=oracle_win_rate_pct,
            is_converged=is_converged,
        )

        return results
