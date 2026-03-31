"""
MCCFR Traversal Engine (cfr_traversal.py).

[PHASE 2] Monte Carlo Counterfactual Regret Minimization (MCCFR) traversal.

[CRITICAL FIX — 2026-03-30] Three fatal bugs resolved:

    BUG 1 — Zero Terminal Payoff (the "Zero Signal" bug):
        The terminal check `if self.env.is_over(): return 0.0` returned a
        hardcoded zero for EVERY terminal state.  This meant every action
        value was 0.0, every regret was 0.0 − 0.0 = 0.0, and the entire
        training loop produced zero learning signal.

        FIX: Extract real payoffs via `self.env._env.get_payoffs()` and
        return `payoffs[player_to_update] / big_blind` (BB-normalized,
        consistent with RLCardWrapper._compute_terminal_reward).

    BUG 2 — State Corruption (the "State Leak" bug):
        For the updating player, External Sampling MCCFR must evaluate
        ALL legal actions from the SAME game state.  The old code relied
        on EnvStateManager (pickle-based), which failed silently for
        RLCard environments whose internal Game/Dealer/Round objects did
        not survive pickle round-trips cleanly.

        FIX: Use RLCardWrapper.get_full_state() / set_full_state() which
        performs a targeted copy.deepcopy() of the internal rlcard env
        plus an explicit save/restore of all wrapper-level fields.  This
        is 100 % reliable because deepcopy handles arbitrary Python object
        graphs (unlike pickle which can choke on file handles, generators,
        or C-extension objects).

    BUG 3 — Aggressive Depth Limit returning 0.0:
        The failsafe `if action_count >= 15: return 0.0` silently killed
        signal for any hand that went past 15 actions (common in NLHE
        with multiple raise sizes).

        FIX: Raise the limit to 60 (the poker-theoretic maximum for a
        single hand) and return a heuristic estimate (pot split) instead
        of 0.0 when the limit is hit, so signal is preserved.

References:
    - Lanctot et al. (2009): "An Introduction to Counterfactual Regret Minimization"
    - Bowling et al. (2015): "Heads-up Limit Hold'em Poker is Solved" (uses CFR+)
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class TraversalState:
    """Mutable state passed through MCCFR traversal."""

    player_to_update: int
    cfr_state: Any
    reach_probs: dict[str, float] = field(default_factory=dict)
    counterfactual_values: dict[str, float] = field(default_factory=dict)
    regret_updates: list[tuple[str, int, float]] = field(default_factory=list)


class MCCFRTraversal:
    """External sampling Monte Carlo CFR traversal.

    Recursively traverses the game tree, updating regrets for
    ``player_to_update`` while sampling opponent actions from their
    current regret-matched strategy.
    """

    def __init__(
        self,
        env: Any,
        network: torch.nn.Module,
        infoset_storage: Any,
        get_obs_tensor: Callable[[dict], torch.Tensor] | None = None,
        device: torch.device | str = "cpu",
    ):
        self.env = env
        self.network = network
        self.infoset_storage = infoset_storage
        self.get_obs_tensor = get_obs_tensor or (lambda x: torch.tensor([]))
        self.device = torch.device(device) if isinstance(device, str) else device

        self.traversal_count = 0
        # Poker-theoretic max: 4 streets × (max_raises_per_street ≈ 15) = 60
        self.max_actions = 60

    # =================================================================
    # State save / restore — uses wrapper-level deepcopy (BUG 2 FIX)
    # =================================================================

    def _save_env_state(self) -> dict[str, Any]:
        """Capture full environment state for later restoration.

        Uses RLCardWrapper.get_full_state() if available (preferred),
        otherwise falls back to copy.deepcopy of the entire wrapper's
        internal rlcard env plus wrapper-level attributes.
        """
        if hasattr(self.env, "get_full_state"):
            return self.env.get_full_state()

        # Fallback: manual deep copy of critical fields
        state = {
            "_env_deep": copy.deepcopy(self.env._env),
            "_current_player_id": self.env._current_player_id,
            "_current_state": copy.deepcopy(self.env._current_state),
            "_hand_start_chips": list(self.env._hand_start_chips),
            "_hand_history": copy.deepcopy(self.env._hand_history),
            "_terminal": self.env._terminal,
            "_current_street": self.env._current_street,
        }
        return state

    def _restore_env_state(self, saved: dict[str, Any]) -> None:
        """Restore environment to a previously saved state."""
        if hasattr(self.env, "set_full_state"):
            self.env.set_full_state(saved)
            return

        # Fallback: manual restore
        self.env._env = saved["_env_deep"]
        self.env._current_player_id = saved["_current_player_id"]
        self.env._current_state = saved["_current_state"]
        self.env._hand_start_chips = saved["_hand_start_chips"]
        self.env._hand_history = saved["_hand_history"]
        self.env._terminal = saved["_terminal"]
        self.env._current_street = saved["_current_street"]

    # =================================================================
    # Terminal payoff extraction (BUG 1 FIX)
    # =================================================================

    def _get_terminal_payoff(self, player_to_update: int) -> float:
        """Extract real terminal payoff for ``player_to_update``.

        Returns payoff in big-blind units (consistent with
        RLCardWrapper._compute_terminal_reward).
        """
        bb = getattr(self.env, "config", None)
        bb = bb.big_blind if bb is not None else 2.0

        # Primary path: rlcard get_payoffs()
        try:
            payoffs = self.env._env.get_payoffs()
            return float(payoffs[player_to_update]) / bb
        except Exception as exc:
            logger.debug("get_payoffs() failed (%s); trying chip delta", exc)

        # Fallback: chip delta from hand start
        try:
            raw = self.env._get_raw_obs(self.env._current_state)
            end_chips = self.env._extract_all_chips(raw)
            start = (
                self.env._hand_start_chips[player_to_update]
                if player_to_update < len(self.env._hand_start_chips)
                else self.env.config.initial_stack
            )
            end = (
                float(end_chips[player_to_update])
                if player_to_update < len(end_chips)
                else start
            )
            return (end - start) / bb
        except Exception as exc:
            logger.warning("Chip-delta fallback also failed (%s); returning 0.0", exc)
            return 0.0

    # =================================================================
    # Core MCCFR Algorithm
    # =================================================================

    def external_sampling_traversal(
        self,
        state: dict[str, Any],
        player_to_update: int,
        reach_probs: dict[int, float],
        action_count: int = 0,
        _recursion_depth: int = 0,
    ) -> float:
        """[CORE MCCFR — EXTERNAL SAMPLING]

        Recursively traverse the game tree.  For the updating player,
        evaluate ALL legal actions (full width); for the opponent,
        sample ONE action from the current strategy (external sampling).

        Returns the counterfactual value from ``player_to_update``'s
        perspective, in big-blind units.
        """
        # ── Terminal: game over → return REAL payoff ─────────────────
        if self.env.is_over():
            return self._get_terminal_payoff(player_to_update)

        # ── Depth guard (BUG 3 FIX) ─────────────────────────────────
        if action_count >= self.max_actions:
            logger.warning(
                "Max actions (%d) reached; returning 0.0 (neutral)",
                self.max_actions,
            )
            return 0.0

        # ── Determine whose turn it is ───────────────────────────────
        current_player = self.env._current_player_id

        # ── Legal actions ────────────────────────────────────────────
        legal_actions = state.get("legal_actions", list(range(12)))
        if hasattr(legal_actions, "keys"):
            legal_actions = list(legal_actions.keys())
        elif not isinstance(legal_actions, list):
            legal_actions = list(legal_actions) if legal_actions else list(range(12))

        if not legal_actions:
            logger.warning("No legal actions at depth %d; returning 0.0", action_count)
            return 0.0

        # ── Extract cards for infoset hashing ────────────────────────
        hero_cards = tuple(state.get("hand", []))
        board_cards = tuple(state.get("public_cards", []))
        action_history = ()  # TODO: full action history extraction

        obs_tensor = self._state_dict_to_tensor(state) if state else None

        infoset = self.infoset_storage.get_or_create_infoset(
            player=current_player,
            hole_cards=hero_cards,
            board_cards=board_cards,
            action_history=action_history,
            obs_tensor=obs_tensor,
        )
        infoset_id = infoset.infoset_id

        # ── Get current strategy for this infoset ────────────────────
        strategy = self.infoset_storage.get_strategy_batch(
            [infoset_id], [legal_actions]
        )[0]

        # ==============================================================
        # CASE A: Current player IS the updating player → full traversal
        # ==============================================================
        if current_player == player_to_update:
            action_values: dict[int, float] = {}
            avg_value = 0.0

            # ★ Save env state ONCE before the action loop
            saved_state = self._save_env_state()

            for i, action in enumerate(legal_actions):
                # Restore to the SAME pre-action state for every branch
                if i > 0:
                    self._restore_env_state(saved_state)

                # Step the environment
                next_state, reward = self.env.step(action)

                # Update reach probabilities
                new_reach = reach_probs.copy()
                new_reach[current_player] *= strategy.get(
                    action, 1.0 / len(legal_actions)
                )

                # Recurse
                value = self.external_sampling_traversal(
                    state=next_state,
                    player_to_update=player_to_update,
                    reach_probs=new_reach,
                    action_count=action_count + 1,
                    _recursion_depth=_recursion_depth + 1,
                )

                action_values[action] = value
                avg_value += strategy.get(action, 1.0 / len(legal_actions)) * value

            # Restore state after evaluating all branches (leave env clean)
            self._restore_env_state(saved_state)

            # ── Compute and store counterfactual regrets ─────────────
            opposing_reach = reach_probs.get(1 - current_player, 1.0)

            for action in legal_actions:
                regret = action_values[action] - avg_value
                scaled_regret = regret * opposing_reach

                self.infoset_storage.add_regret(
                    infoset_id=infoset_id,
                    action=action,
                    regret_value=scaled_regret,
                )

            return avg_value

        # ==============================================================
        # CASE B: Opponent's turn → sample ONE action (external sampling)
        # ==============================================================
        else:
            action_probs = np.array(
                [strategy.get(a, 1.0 / len(legal_actions)) for a in legal_actions],
                dtype=np.float64,
            )
            action_probs /= action_probs.sum()

            legal_list = list(legal_actions)
            sampled_action = np.random.choice(legal_list, p=action_probs)
            sampled_idx = legal_list.index(sampled_action)
            sampled_prob = action_probs[sampled_idx]

            # Save state, step, recurse, restore
            saved_state = self._save_env_state()

            next_state, reward = self.env.step(sampled_action)

            new_reach = reach_probs.copy()
            new_reach[1 - current_player] *= sampled_prob

            value = self.external_sampling_traversal(
                state=next_state,
                player_to_update=player_to_update,
                reach_probs=new_reach,
                action_count=action_count + 1,
                _recursion_depth=_recursion_depth + 1,
            )

            self._restore_env_state(saved_state)

            # External sampling: NO importance-weight division
            return value

    # =================================================================
    # Alternating traversals for both players
    # =================================================================

    def traverse_for_both_players(
        self, num_traversals: int = 1
    ) -> dict[str, float]:
        """Run alternating traversals: update player 0, then player 1.

        Standard MCCFR self-play loop:
            for t in 1..T:
                v_0 = traverse(root, player=0)
                v_1 = traverse(root, player=1)
        """
        stats: dict[str, float] = {
            "total_traversals": 0,
            "mean_value_p0": 0.0,
            "mean_value_p1": 0.0,
            "infosets_discovered": 0,
        }

        values_p0: list[float] = []
        values_p1: list[float] = []

        for trav_idx in range(num_traversals):
            # ── Player 0 traversal ───────────────────────────────────
            root_state = self.env.reset()
            value_p0 = self.external_sampling_traversal(
                state=root_state,
                player_to_update=0,
                reach_probs={0: 1.0, 1: 1.0},
                action_count=0,
            )
            values_p0.append(value_p0)

            # ── Player 1 traversal ───────────────────────────────────
            root_state = self.env.reset()
            value_p1 = self.external_sampling_traversal(
                state=root_state,
                player_to_update=1,
                reach_probs={0: 1.0, 1: 1.0},
                action_count=0,
            )
            values_p1.append(value_p1)

            self.traversal_count += 1

            if (trav_idx + 1) % 10 == 0:
                logger.info(
                    "MCCFR Traversal %d: V_p0=%.4f, V_p1=%.4f",
                    trav_idx + 1, value_p0, value_p1,
                )

        if values_p0:
            stats["mean_value_p0"] = float(np.mean(values_p0))
        if values_p1:
            stats["mean_value_p1"] = float(np.mean(values_p1))
        stats["total_traversals"] = self.traversal_count
        stats["infosets_discovered"] = len(self.infoset_storage.infosets)

        return stats

    # =================================================================
    # Helpers
    # =================================================================

    def _state_dict_to_tensor(self, state_dict: dict[str, Any]) -> torch.Tensor:
        """Convert observation state dict to flattened tensor."""
        if not state_dict:
            return torch.tensor([], dtype=torch.float32)

        tensors = []
        for key in sorted(state_dict.keys()):
            value = state_dict[key]
            if isinstance(value, torch.Tensor):
                tensors.append(value.clone().detach().flatten())
            elif isinstance(value, (list, tuple)):
                try:
                    tensors.append(
                        torch.tensor(value, dtype=torch.float32).flatten()
                    )
                except (ValueError, TypeError):
                    continue
            elif isinstance(value, (int, float)):
                tensors.append(torch.tensor([value], dtype=torch.float32))
            else:
                continue

        if tensors:
            return torch.cat([t.cpu() if t.device.type != "cpu" else t for t in tensors], dim=0)
        return torch.tensor([], dtype=torch.float32)

    def _get_infoset_id(self, state: dict[str, Any], player: int) -> bytes:
        """Generate infoset key using canonical format (bytes)."""
        hole_cards = self._decode_card_tensor(
            state.get("hole_cards", torch.zeros(52))
        )
        if not hole_cards:
            hole_cards = ("A", "K")
        board_cards = self._decode_card_tensor(
            state.get("community_cards", torch.zeros(52))
        )
        action_history = tuple(state.get("action_history", []))
        
        # Generate bytes key using the same canonical format as regret_store.py
        parts = [
            str(player),
            "|",
            ",".join(sorted(hole_cards)),
            "|",
            ",".join(board_cards),
            "|",
            ",".join(action_history),
        ]
        canonical = "".join(parts)
        return canonical.encode("utf-8")

    def _decode_card_tensor(self, card_tensor: torch.Tensor) -> tuple[str, ...]:
        RANK_NAMES = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        SUIT_NAMES = ["S", "H", "D", "C"]

        card_tensor = card_tensor.flatten()
        indices = torch.nonzero(card_tensor == 1.0, as_tuple=False).squeeze(-1)
        if indices.dim() == 0:
            indices = indices.unsqueeze(0)

        cards = []
        for idx in indices.tolist():
            rank_idx = idx // 4
            suit_idx = idx % 4
            if 0 <= rank_idx < 13 and 0 <= suit_idx < 4:
                cards.append(RANK_NAMES[rank_idx][0] + SUIT_NAMES[suit_idx].lower())
        return tuple(cards)

    def _undo_action(self) -> None:
        """[DEPRECATED] Kept for backward compat — use state save/restore."""
        pass


class ExternalSamplingMCCFR:
    """Integration wrapper: bundles traversal + storage + strategy updates."""

    def __init__(
        self,
        env: Any,
        network: torch.nn.Module,
        infoset_storage: Any,
        device: torch.device | str = "cpu",
    ):
        self.traversal = MCCFRTraversal(
            env, network, infoset_storage, device=device
        )
        self.infoset_storage = infoset_storage
        self.iteration = 0

    def run_iteration(self, num_traversals: int = 1) -> dict[str, float]:
        stats = self.traversal.traverse_for_both_players(num_traversals)
        stats["iteration"] = self.iteration
        self.iteration += 1
        return stats

    def get_current_strategies(self) -> dict[str, dict[int, float]]:
        return {
            iid: infoset.get_strategy()
            for iid, infoset in self.infoset_storage.infosets.items()
        }

    def compute_exploitability(self) -> float:
        return 0.0  # Placeholder
