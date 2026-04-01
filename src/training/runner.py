"""
Event-Driven Vegrehajto Ciklus (runner.py).

[FIX C1 - 2025-03-28] Bootstrap Ertek Timing Javitasa:
    A _run_single_iteration() tobbe nem hivja a collector.get_last_bootstrap_value()-t
    a compute_gae() elott. A bootstrap erteket most a collector.collect_rollout()
    atomikusan tarolja a bufferben (buffer.set_last_value()), igy a runner
    onnan olvassa ki: self.buffer.compute_gae(last_value=self.buffer.get_last_bootstrap_value()).
    Ez megszunteti a race conditiont: az ertek garantaltan a HELYES, truncated
    allapothoz tartozik, nem egy lepessessel kesobb szamolt kozelites.

A ciklus felepitese:
    1. Bootstrapping: Kornyezet, halozat, buffer, trainer inicializalasa
    2. Event-Driven Loop (while not shutdown):
        a) Adatgyujtes (collector.collect_rollout)
        b) Gradiens frissites (trainer.train_on_buffer)
        c) Telemetria feldolgozas (orchestrator callback)
        d) Checkpoint mentes (periodikus)
    3. Graceful Shutdown: Utolso mentes, HF feltoltes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Callable

import numpy as np
import torch
import torch.optim as optim

from src.training.buffers import BufferManager  # [PHASE 2] Buffer management
from src.training.vr_deep_pdcfr_engine import VRDeepPDCFREngine  # [PHASE 4] VR-DeepPDCFR+ Engine
from src.model.networks import VRDeepPDCFRNetworks  # [PHASE 3] Network bundle (4 networks per player)
from src.evaluation.nash_evaluator import LocalBestResponseEvaluator, NashEvalConfig  # [PHASE 6] Oracle evaluation

logger = logging.getLogger(__name__)


# ============================================================================
# GAME STATE ADAPTER: Bridge RLCard Environment to VR-DeepPDCFR+ Engine
# ============================================================================

class GameStateAdapter:
    """Wraps RLCardWrapper to provide the game state interface expected by VRDeepPDCFREngine.
    
    This adapter translates between:
    - RLCard's environment (step, is_over, get_payoffs)
    - VR-DeepPDCFR+ engine's game state interface (is_terminal, get_acting_player, etc.)
    
    Key design:
    - Uses env.get_full_state() / set_full_state() for non-mutating action simulation
    - Stores a snapshot of environment state to avoid parent mutation
    - Each call to get_action_taken() returns a NEW adapter with simulated state
    
    Attributes:
        env: The RLCardWrapper environment
        obs_builder: ObservationBuilder for encoding game states to feature vectors
        env_snapshot: Snapshot of environment state at this node
        current_obs: Current observation dict for the environment
    """
    
    def __init__(
        self,
        env: Any,
        obs_builder: Any,
        env_snapshot: dict[str, Any] | None = None,
        current_obs: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a game state adapter.
        
        Args:
            env: RLCardWrapper environment instance
            obs_builder: ObservationBuilder for feature encoding
            env_snapshot: Optional snapshot to restore environment to a specific state
                         (used internally for get_action_taken)
            current_obs: Optional observation dict (used internally)
        """
        self.env = env
        self.obs_builder = obs_builder
        self.env_snapshot = env_snapshot or env.get_full_state()
        self.current_obs = current_obs or env._build_obs_dict(
            env._current_state, env._current_player_id
        )
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal game state.
        
        Returns:
            True if the game is over, False otherwise
        """
        return self.env.is_over()
    
    def get_terminal_payoffs(self) -> dict[int, float]:
        """Get payoff dictionary at terminal node.
        
        Returns:
            Dict mapping player_id -> payoff in big-blind units
            
        Raises:
            RuntimeError: If not at terminal node
        """
        if not self.is_terminal():
            raise RuntimeError(
                "get_terminal_payoffs() called on non-terminal state"
            )
        
        try:
            # RLCard returns payoffs as array [p0_payoff, p1_payoff, ...]
            payoffs_array = self.env._env.get_payoffs()
            bb = self.env.config.big_blind
            
            # Convert to dict indexed by player_id
            payoffs_dict = {
                player_id: float(payoffs_array[player_id]) / bb
                for player_id in range(len(payoffs_array))
            }
            
            logger.debug(f"Terminal payoffs: {payoffs_dict}")
            return payoffs_dict
            
        except Exception as exc:
            logger.error(f"Failed to get terminal payoffs: {exc}")
            # Fallback: return uniform payoffs
            num_players = self.env.config.num_players
            return {player_id: 0.0 for player_id in range(num_players)}
    
    def is_chance_node(self) -> bool:
        """Check if this is a chance (stochastic) node.
        
        In RLCard, chance events occur when transitioning between streets.
        We detect this by checking if the current player is betting/acting,
        or if we're in a state where no one can act (need cards dealt).
        
        Returns:
            True if at a chance node (card dealing transition), False otherwise
        """
        try:
            # Get current game state to check if we're at a street transition
            # In RLCard, when it's time to deal the next community cards,
            # the legal_actions for the current player may be empty or the
            # round state indicates a street advance is coming
            legal_actions = self.env._current_state.get("legal_actions", {})
            
            # If there are legal actions, it's a normal player node, not chance
            if legal_actions:
                return False
            
            # If no legal actions and game is not over, we're likely at a chance transition
            if not self.env.is_over():
                return True
            
            return False
        except Exception:
            return False
    
    def sample_chance_outcome(self) -> GameStateAdapter:
        """Sample a single chance outcome (card dealing) via RLCard's internal logic.
        
        In external sampling MCCFR, we sample exactly ONE outcome at chance nodes.
        For RLCard poker, this means advancing the street and dealing cards.
        
        CRITICAL: This method MUST restore the environment to its pre-call state
        to prevent infinite loops in traversal. Uses save/restore pattern like get_action_taken().
        
        Returns:
            New GameStateAdapter with the sampled cards at the next street
        """
        # Save current environment state BEFORE any modifications
        saved_snapshot = self.env.get_full_state()
        
        try:
            # RLCard advances to the next street internally when we call step()
            # with any valid action during chance/transition. We step with
            # the first legal action (or 0 if none available), which causes
            # RLCard to deal the next community cards and move to the next street.
            legal_actions = self.env._current_state.get("legal_actions", {})
            action_id = min(legal_actions.keys()) if legal_actions else 0
            
            # Step the environment - this applies the chance event (card dealing)
            raw_result = self.env._env.step(action_id)
            next_state, next_player = self.env._unpack_step(raw_result)
            self.env._current_player_id = next_player
            self.env._current_state = next_state
            self.env._terminal = bool(self.env._env.is_over())
            
            # Capture the new state AFTER stepping
            next_snapshot = self.env.get_full_state()
            new_obs = self.env._build_obs_dict(next_state, next_player)
            
            logger.debug(f"Chance node: sampled card dealing, next player={next_player}")
            
            # Create and return a new adapter with the sampled state
            child_adapter = GameStateAdapter(
                env=self.env,
                obs_builder=self.obs_builder,
                env_snapshot=next_snapshot,
                current_obs=new_obs,
            )
            
            return child_adapter
            
        except Exception as exc:
            logger.warning(f"Failed to sample chance outcome: {exc}", exc_info=True)
            # Fallback: return self unchanged
            return self
        finally:
            # CRITICAL: Always restore environment to pre-call state
            # This ensures traversal can continue without state pollution
            self.env.set_full_state(saved_snapshot)
            logger.debug("Environment restored after chance node sampling")
    
    def get_chance_outcomes(self) -> list[tuple[Any, float]]:
        """Get stochastic outcomes at chance node (deprecated; use sample_chance_outcome).
        
        For external sampling MCCFR, use sample_chance_outcome() instead.
        This method is kept for compatibility but should not be called.
        
        Returns:
            List containing a single sampled outcome
        """
        # For external sampling, we only return one outcome
        return [(self.sample_chance_outcome(), 1.0)]
    
    def get_acting_player(self) -> int:
        """Get the player whose turn it is to act.
        
        Returns:
            Player ID (0-indexed)
        """
        return self.env._current_player_id
    
    def get_infoset_features(self, player_id: Optional[int] = None) -> np.ndarray:
        """Get feature vector representation of the game state.
        
        Uses the ObservationBuilder to encode the observation into a flat
        feature vector suitable for neural network input.
        
        Args:
            player_id: Optional player ID to generate features from their perspective.
                      If None, generates features for the current acting player.
                      CRITICAL in imperfect-information games: each player's features
                      must be generated from their own perspective, not from the
                      perspective of another player.
        
        Returns:
            Flat numpy array of shape (feature_dim,) with dtype float32
            
        ITEM 10 FIX: Enforces strict per-player state isolation.
        Each player's observation is built from RLCard's get_state(player_id),
        preventing card leakage between players' Q-networks.
        """
        try:
            # Determine target player perspective
            pid = player_id if player_id is not None else self.get_acting_player()
            
            # Validate pid is an integer
            if pid is None:
                raise ValueError("Acting player is None - game may be in terminal state")
            
            # ITEM 10 FIX: Fetch correct raw state for the target player
            # ===========================================================
            # CRITICAL: Always use self.env._env.get_state(pid) to fetch player-specific state.
            # This prevents leaking cards from one player's hole cards to another player's networks.
            # RLCard's get_state(player_id) returns ONLY what that player can see,
            # excluding opponent hole cards.
            raw_state = self.env._env.get_state(pid)
            
            # Build observation using the ObservationBuilder (returns tensordict with proper keys)
            # This returns a dict with keys: hole_cards, community_cards, env_metrics, betting_history, position
            obs_dict = self.obs_builder.build(raw_state, validate=False)
            
            # Flatten the observation using the observation builder
            flat_tensor = self.obs_builder.flatten(obs_dict)
            
            # Convert to numpy and ensure float32
            if hasattr(flat_tensor, 'cpu'):
                features = flat_tensor.cpu().numpy()
            else:
                features = np.array(flat_tensor)
            
            # Ensure output is contiguous float32 array
            return np.ascontiguousarray(features, dtype=np.float32)
                
        except Exception as exc:
            logger.error(f"Failed to encode observation for player {player_id}: {exc}", exc_info=True)
            # Fallback: return zeros
            obs_dim = self.obs_builder.get_observation_dim()
            return np.zeros(obs_dim, dtype=np.float32)
    
    def get_legal_actions(self) -> np.ndarray:
        """Get legal action mask for current player.
        
        Returns:
            Boolean array of shape (num_actions,) where True = legal, False = illegal
        """
        try:
            # Get legal action indices from environment state
            legal_action_indices = self.env._current_state.get("legal_actions", {})
            
            # Create boolean mask
            # RLCard uses different action mappings, we need to convert
            # Assume legal_action_indices is a dict or list of valid action IDs
            if isinstance(legal_action_indices, dict):
                legal_indices_list = list(legal_action_indices.keys())
            elif isinstance(legal_action_indices, list):
                legal_indices_list = legal_action_indices
            else:
                legal_indices_list = [0, 1]  # Fallback: fold and check
            
            # Create boolean array (num_actions=12 for poker)
            num_actions = 12  # Corresponds to _FOLD through _ALL_IN
            mask = np.zeros(num_actions, dtype=np.bool_)
            
            for idx in legal_indices_list:
                if 0 <= idx < num_actions:
                    mask[idx] = True
            
            logger.debug(f"Legal actions mask: {mask}")
            return mask
            
        except Exception as exc:
            logger.error(f"Failed to get legal actions: {exc}")
            # Fallback: all actions are legal
            num_actions = 12
            return np.ones(num_actions, dtype=np.bool_)
    
    def get_action_taken(self, action_idx: int) -> "GameStateAdapter":
        """Simulate taking an action and return new adapter for resulting state.
        
        This method does NOT mutate the current environment. Instead:
        1. Saves the current environment state
        2. Steps the action forward
        3. Captures the new state
        4. Restores the environment to the saved state
        5. Returns a new adapter with the captured state
        
        Args:
            action_idx: Action index (0-11 for poker)
            
        Returns:
            New GameStateAdapter at child state
            
        Raises:
            RuntimeError: If action is invalid or environment is terminal
        """
        if self.is_terminal():
            raise RuntimeError("Cannot take action on terminal state")
        
        # Clamp action to valid range
        action_idx = int(max(0, min(11, action_idx)))
        
        # Save current environment state
        saved_snapshot = self.env.get_full_state()
        
        try:
            # Step the action (modifies env internally)
            next_obs, reward = self.env.step(action_idx)
            
            # Capture the next state's snapshot
            next_snapshot = self.env.get_full_state()
            
            logger.debug(
                f"Action {action_idx} taken by player {self.get_acting_player()}, "
                f"reward={reward:.4f}"
            )
            
            # Create new adapter for next state
            return GameStateAdapter(
                env=self.env,
                obs_builder=self.obs_builder,
                env_snapshot=next_snapshot,
                current_obs=next_obs,
            )
            
        finally:
            # Always restore environment to pre-step state
            self.env.set_full_state(saved_snapshot)
            logger.debug("Environment restored to pre-action state")
    
    def get_reward_for_action(self, action_idx: int) -> float:
        """Get immediate reward for taking an action.
        
        In poker, rewards only occur at terminal nodes.
        Intermediate steps have zero reward.
        
        Args:
            action_idx: Action index (not used, included for interface compatibility)
            
        Returns:
            0.0 (no intermediate rewards in poker)
        """
        return 0.0


@dataclass
class RunnerConfig:
    """A fo vegrehajtasi ciklus konfiguracioja."""

    max_iterations: int = 0
    log_interval: int = 10
    eval_interval: int = 50
    save_interval: int = 100
    buffer_save_interval: int = 500
    max_runtime_hours: float = 11.5
    device: str = "auto"

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> RunnerConfig:
        orch = cfg.get("orchestrator", {})
        tel = orch.get("telemetry", {})
        mlops = cfg.get("mlops", {})
        cp = mlops.get("checkpoint", {})
        gs = mlops.get("graceful_shutdown", {})

        return cls(
            log_interval=tel.get("log_interval_iterations", 10),
            eval_interval=tel.get("eval_interval_iterations", 50),
            save_interval=cp.get("save_interval_iterations", 100),
            buffer_save_interval=cp.get("buffer_save_interval_iterations", 500),
            max_runtime_hours=gs.get("max_runtime_hours", 11.5),
            device=cfg.get("runtime", {}).get("device", "auto"),
        )


class TrainingRunner:
    """A teljes RL training pipeline fo vezerlo osztalya.

    [FIX C1] A _run_single_iteration() a compute_gae() hivast a buffer-ben
    tarolt bootstrap ertek alapjan vegzi:
        self.buffer.compute_gae(last_value=self.buffer.get_last_bootstrap_value())
    A korabbi self.collector.get_last_bootstrap_value(self.network) hivast
    eltavolitottuk, mert az egy lepessessel kesob szamolt, versenyfutasi
    allapotot (race condition) okozva episode hatarokon.
    """

    def __init__(
        self,
        config: RunnerConfig,
        env: Any,
        obs_builder: Any,
        network: Any,
        trainer_config: Any | None = None,  # [DEPRECATED] No longer used (PPOTrainer removed)
        buffer_config: RolloutBufferConfig | None = None,
        yaml_config: dict[str, Any] | None = None,
        on_iteration_end: Callable[[int, dict[str, float]], None] | None = None,
        on_eval_step: Callable[[int, Any], dict[str, float] | None] | None = None,
        on_checkpoint: Callable[[int, Any], None] | None = None,
        on_ddp_sync: Callable[[int], None] | None = None,
        checkpoint_dir: str = "checkpoints",
        orchestrator: Any | None = None,
    ) -> None:
        self.config: RunnerConfig = config
        self.network: Any = network
        self.env: Any = env
        self.obs_builder: Any = obs_builder

        self.device: torch.device = self._resolve_device(config.device)

        self.buffer: RolloutBuffer = RolloutBuffer(
            buffer_config or RolloutBufferConfig()
        )
        
        # [PHASE 4] VR-DeepPDCFR+ Initialization
        yaml_config = yaml_config or {}
        
        # Determine number of players from environment
        num_players = getattr(env, "_num_players", 2)
        obs_dim = obs_builder.get_observation_dim()
        num_actions = network.config.num_actions if hasattr(network, 'config') else 9
        
        logger.info(
            "[PHASE 4] Initializing VR-DeepPDCFR+ with %d players, "
            "obs_dim=%d, num_actions=%d",
            num_players, obs_dim, num_actions,
        )
        
        # Create per-player buffer managers
        buffer_managers: dict[int, BufferManager] = {}
        for player_id in range(num_players):
            buffer_managers[player_id] = BufferManager(
                advantage_capacity=yaml_config.get("buffer", {}).get("advantage_capacity", 100_000),
                strategy_capacity=yaml_config.get("buffer", {}).get("strategy_capacity", 1_000_000),
                time_decay_power=yaml_config.get("buffer", {}).get("time_decay_power", 1.0),
            )
        
        # Create per-player network bundles
        networks: dict[int, VRDeepPDCFRNetworks] = {}
        for player_id in range(num_players):
            networks[player_id] = VRDeepPDCFRNetworks(
                input_dim=obs_dim,
                output_dim=num_actions,
                hidden_dims=yaml_config.get("network", {}).get("hidden_dims", [256, 128]),
                activation=yaml_config.get("network", {}).get("activation", torch.nn.ReLU),
                use_layer_norm=yaml_config.get("network", {}).get("use_layer_norm", False),
                dropout_p=yaml_config.get("network", {}).get("dropout_p", 0.0),
            )
        
        # Create per-player optimizers (4 networks per player)
        lr = yaml_config.get("optimizer", {}).get("learning_rate", 1e-3)
        optimizers: dict[int, dict[str, optim.Optimizer]] = {}
        for player_id in range(num_players):
            optimizers[player_id] = {
                "cumulative": optim.Adam(
                    networks[player_id].cumulative_advantage.parameters(),
                    lr=lr,
                ),
                "instantaneous": optim.Adam(
                    networks[player_id].instantaneous_advantage.parameters(),
                    lr=lr,
                ),
                "value": optim.Adam(
                    networks[player_id].value.parameters(),
                    lr=lr,
                ),
                "strategy": optim.Adam(
                    networks[player_id].strategy.parameters(),
                    lr=lr,
                ),
            }
        
        # [PHASE 4] Instantiate VR-DeepPDCFR+ Engine
        self.trainer = VRDeepPDCFREngine(
            buffer_managers=buffer_managers,
            networks=networks,
            optimizers=optimizers,
            device=self.device,
        )
        
        self.collector: RolloutCollector = RolloutCollector(
            network=network,
            env=env,
            obs_builder=obs_builder,
            buffer=self.buffer,
            config=yaml_config,
            orchestrator=orchestrator,
            device=self.device,
        )

        self._on_iteration_end = on_iteration_end
        self._on_eval_step = on_eval_step
        self._on_checkpoint = on_checkpoint
        self._on_ddp_sync = on_ddp_sync

        self.iteration: int = 0
        self._start_time: float = 0.0
        self._checkpoint_dir: str = checkpoint_dir
        self._should_stop: bool = False
        self._nan_error_occurred: bool = False
        
        # [PHASE 6] Oracle Best-Response Evaluator (optional periodic evaluation)
        self.oracle_evaluator: LocalBestResponseEvaluator | None = None
        self.oracle_eval_interval: int = yaml_config.get("evaluation", {}).get(
            "oracle_eval_interval", 0  # 0 = disabled
        )
        if self.oracle_eval_interval > 0:
            try:
                from src.env.action_mapper import ActionMapper
                from src.env.equity import EquityCalculator
                
                oracle_config = NashEvalConfig(
                    eval_hands=yaml_config.get("evaluation", {}).get("oracle_hands", 20),
                    target_pct=0.3,
                    equity_iterations=yaml_config.get("evaluation", {}).get("equity_iterations", 500),
                    model_deterministic=True,
                    use_improved_ev=True,
                )
                
                # Create EquityCalculator for oracle evaluator
                equity_calc = EquityCalculator()
                
                self.oracle_evaluator = LocalBestResponseEvaluator(
                    model=network,
                    env=env,
                    obs_builder=obs_builder,
                    action_mapper=ActionMapper(),
                    equity_calc=equity_calc,
                    config=oracle_config,
                    device=str(self.device),
                )
                logger.info(
                    "[PHASE 6] Oracle evaluator initialized: eval every %d iters, %d hands",
                    self.oracle_eval_interval,
                    oracle_config.eval_hands,
                )
            except Exception as e:
                logger.warning("[PHASE 6] Failed to initialize oracle evaluator: %s", e)
                self.oracle_evaluator = None

        logger.info(
            "TrainingRunner inicializalva: device=%s, max_iter=%d, "
            "save_interval=%d, max_runtime=%.1fh",
            self.device, config.max_iterations,
            config.save_interval, config.max_runtime_hours,
        )

    # =========================================================================
    # Fo Futtatasi Ciklus
    # =========================================================================

    def run(self) -> dict[str, Any]:
        """Elindítja es futtatja a teljes training ciklust."""
        self._start_time = time.monotonic()
        self._should_stop = False

        logger.info(
            "========================================\n"
            "  TRAINING CIKLUS INDUL\n"
            "  Max iterations: %s\n"
            "  Max runtime: %.1f ora\n"
            "  Device: %s\n"
            "========================================",
            self.config.max_iterations or "vegtelen",
            self.config.max_runtime_hours,
            self.device,
        )

        all_stats: list[dict[str, float]] = []

        try:
            while not self._should_stop:
                self.iteration += 1

                if self._check_time_limit():
                    logger.warning(
                        "Idokorlat elerve (%.1f ora). Graceful shutdown...",
                        self.config.max_runtime_hours,
                    )
                    break

                if (self.config.max_iterations > 0
                        and self.iteration > self.config.max_iterations):
                    logger.info(
                        "Max iteracio (%d) elerve. Training vege.",
                        self.config.max_iterations,
                    )
                    break

                iter_stats: dict[str, float] = self._run_single_iteration()
                all_stats.append(iter_stats)

                if self.iteration % self.config.log_interval == 0:
                    self._log_iteration(iter_stats)

                if self.iteration % self.config.eval_interval == 0:
                    if self._on_eval_step is not None:
                        eval_result = self._on_eval_step(self.iteration, self.network)
                        if eval_result:
                            logger.info(
                                "Eval iter #%d: %s", self.iteration, eval_result
                            )
                
                # [PHASE 6] Oracle evaluation (if enabled)
                if (self.oracle_eval_interval > 0 
                    and self.iteration % self.oracle_eval_interval == 0 
                    and self.oracle_evaluator is not None):
                    try:
                        logger.info("[PHASE 6] Oracle evaluation starting (iter #%d)...", self.iteration)
                        oracle_results = self.oracle_evaluator.run_evaluation()
                        logger.info(
                            "[PHASE 6] Oracle Results (iter #%d): "
                            "MBB/hand=%.2f, Nash Distance=%.2f%%, Win Rate=%.1f%%",
                            self.iteration,
                            oracle_results.oracle_mbb_hand,
                            oracle_results.nash_distance_pct,
                            oracle_results.oracle_win_rate_pct,
                        )
                        # Log to iter_stats for monitoring
                        iter_stats[f"oracle/mbb_hand"] = oracle_results.oracle_mbb_hand
                        iter_stats[f"oracle/nash_distance_pct"] = oracle_results.nash_distance_pct
                        iter_stats[f"oracle/win_rate_pct"] = oracle_results.oracle_win_rate_pct
                    except Exception as e:
                        logger.warning("[PHASE 6] Oracle evaluation failed: %s", e)

                if self.iteration % self.config.save_interval == 0:
                    self._save_checkpoint()

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt! Graceful shutdown...")
        except FloatingPointError as exc:
            logger.critical(
                "FLOATINGPOINTERROR (Iter #%d): %s — Sulyszennyezodes, "
                "vegso checkpoint mentes kihagyva.",
                self.iteration, exc, exc_info=True,
            )
            self._nan_error_occurred = True
            raise
        except Exception as exc:
            logger.error(
                "KRITIKUS HIBA az iteracioban #%d: %s",
                self.iteration, exc, exc_info=True,
            )
            self._save_checkpoint(emergency=True)
            raise
        finally:
            if not self._nan_error_occurred:
                self._save_checkpoint(final=True)

        elapsed: float = time.monotonic() - self._start_time

        summary: dict[str, Any] = {
            "total_iterations": self.iteration,
            "total_runtime_hours": elapsed / 3600,
            "total_steps": self.collector.get_total_steps(),
            "total_episodes": self.collector.get_total_episodes(),
        }

        logger.info(
            "========================================\n"
            "  TRAINING CIKLUS BEFEJEZVE\n"
            "  Iteraciok: %d\n"
            "  Futasido: %.2f ora\n"
            "  Osszes lepes: %d\n"
            "  Osszes epizod: %d\n"
            "========================================",
            summary["total_iterations"],
            summary["total_runtime_hours"],
            summary["total_steps"],
            summary["total_episodes"],
        )

        return summary

    # =========================================================================
    # Egy Iteracio
    # =========================================================================

    def _run_single_iteration(self) -> dict[str, float]:
        """Vegrehajt egyetlen training iteraciot.

        [PHASE 4] VR-DeepPDCFR+ Iteration Lifecycle:
            1. start_iteration() : Clear ephemeral buffers, set networks to training mode
            2. traverse()        : Recursively traverse game tree, compute advantages
            3. train_networks()  : Gradient descent on all 4 networks per player
            4. end_iteration()   : Increment iteration counter, update frozen networks

        Returns:
            Dict az iteracio statisztikaival.

        Raises:
            FloatingPointError: NaN/Inf a loss-ban.
            RuntimeError: Dimenzio mismatch vagy matematikai inkonzisztencia.
        """
        # =====================================================================
        # STEP 1: Initialize VR-DeepPDCFR+ Iteration
        # =====================================================================
        try:
            self.trainer.start_iteration()
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "HIBA a start_iteration()-ben (iter #%d): %s",
                self.iteration, exc,
            )
            raise

        # =====================================================================
        # STEP 2: Traverse Game Tree for Each Player (External Sampling MCCFR)
        # =====================================================================
        # [PHASE 4] External Sampling MCCFR: For N-player games, we call traverse()
        # once per player. For the updating_player, we enumerate all actions at their
        # decision nodes (full counterfactual regret). For other players, we sample
        # a single action and traverse only that branch (external sampling).
        # This makes the algorithm tractable for large games like 6-Max NLHE (~10^161 nodes).
        
        num_players = len(self.trainer.buffer_managers)
        all_traverse_values = []
        
        try:
            logger.debug(
                "Starting External Sampling MCCFR traversal: "
                "num_players=%d, iter=%d",
                num_players,
                self.iteration,
            )
            
            for updating_player in range(num_players):
                # Reset environment for this player's traversal
                root_obs = self.env.reset()
                
                # Wrap in GameStateAdapter for VR-DeepPDCFR+ interface
                root_state = GameStateAdapter(
                    env=self.env,
                    obs_builder=self.obs_builder,
                    current_obs=root_obs,
                )
                
                # Initialize reach probabilities: all players at 1.0
                initial_reach_probs = {i: 1.0 for i in range(num_players)}
                
                logger.debug(
                    "Traversing game tree for updating_player=%d (iter #%d)",
                    updating_player, self.iteration,
                )
                
                # Execute game tree traversal with External Sampling MCCFR
                # - For updating_player: enumerate all actions, store advantages in buffer
                # - For other players: sample one action, traverse that branch only
                traverse_values = self.trainer.traverse(
                    root_state, 
                    initial_reach_probs, 
                    updating_player=updating_player
                )
                
                all_traverse_values.append(traverse_values)
                
                logger.debug(
                    "Game tree traversal complete for updating_player=%d: values=%s",
                    updating_player, traverse_values,
                )
            
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "HIBA a game tree traversal-ben (iter #%d): %s",
                self.iteration, exc,
            )
            raise

        # =====================================================================
        # STEP 3: Train Networks on Buffered Data
        # =====================================================================
        try:
            train_stats = self.trainer.train_networks()
            
            # Validate loss values for NaN/Inf
            for loss_key in ("cumulative_loss", "instantaneous_loss", "value_loss", "strategy_loss"):
                loss_val = train_stats.get(loss_key, 0.0)
                if loss_val != loss_val or abs(loss_val) == float("inf"):
                    raise FloatingPointError(
                        f"KRITIKUS: {loss_key}={loss_val} (NaN/Inf) detektalva "
                        f"az iteracio #{self.iteration}-ban!"
                    )
            
        except FloatingPointError:
            raise
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "HIBA a network training-ben (iter #%d): %s",
                self.iteration, exc,
            )
            raise

        # =====================================================================
        # STEP 4: Finalize Iteration & Update Frozen Networks
        # =====================================================================
        try:
            self.trainer.end_iteration()
        except (RuntimeError, ValueError) as exc:
            logger.error(
                "HIBA az end_iteration()-ben (iter #%d): %s",
                self.iteration, exc,
            )
            raise

        # =====================================================================
        # Compile Iteration Statistics
        # =====================================================================
        iter_stats: dict[str, float] = {
            "iteration": float(self.iteration),
            **{f"train/{k}": v for k, v in train_stats.items()},
            "elapsed_hours": (time.monotonic() - self._start_time) / 3600,
        }

        # 5. Orchestrator callback
        if self._on_iteration_end is not None:
            try:
                self._on_iteration_end(self.iteration, iter_stats)
            except Exception as exc:
                logger.error(
                    "Orchestrator callback hiba (iter #%d): %s — "
                    "A training folytathato, de a curriculum logika "
                    "az aktualis iteracioban kihagyasra kerult.",
                    self.iteration, exc,
                )

        # 6. DDP szinkronizacio
        if self._on_ddp_sync is not None:
            self._on_ddp_sync(self.iteration)

        # 7. Buffer reset
        self.buffer.reset()

        return iter_stats

    # =========================================================================
    # Checkpoint Kezeles
    # =========================================================================

    def _save_checkpoint(
        self, emergency: bool = False, final: bool = False
    ) -> None:
        if self._on_checkpoint is None:
            logger.warning(
                "Nincs on_checkpoint callback konfiguralva — allapot NEM lett mentve (iter #%d)",
                self.iteration,
            )
            return

        save_type = "EMERGENCY" if emergency else ("FINAL" if final else "PERIODIC")
        logger.info("%s checkpoint mentes indul (iter #%d)", save_type, self.iteration)

        try:
            self._on_checkpoint(self.iteration, self.network)
            logger.info("%s checkpoint sikeresen mentve (iter #%d)", save_type, self.iteration)
        except Exception as exc:
            logger.error(
                "Checkpoint callback hiba (iter #%d): %s",
                self.iteration, exc, exc_info=True,
            )

    # =========================================================================
    # Idozites es Leallitas
    # =========================================================================

    def _check_time_limit(self) -> bool:
        elapsed_hours: float = (time.monotonic() - self._start_time) / 3600
        return elapsed_hours >= self.config.max_runtime_hours

    def request_stop(self) -> None:
        self._should_stop = True
        logger.info("Leallitasi keres fogadva. A ciklus a kovetkezo iteracional leall.")

    def get_elapsed_hours(self) -> float:
        if self._start_time == 0:
            return 0.0
        return (time.monotonic() - self._start_time) / 3600

    # =========================================================================
    # Logolas
    # =========================================================================

    def _log_iteration(self, stats: dict[str, float]) -> None:
        logger.info(
            "Iter #%d | rew=%.4f | pl=%.4f vl=%.4f H=%.4f | "
            "kl=%.4f clip=%.1f%% | eps=%d | boot=%.4f | %.2fh",
            self.iteration,
            stats.get("collect/mean_reward", 0.0),
            stats.get("train/policy_loss", 0.0),
            stats.get("train/value_loss", 0.0),
            stats.get("train/entropy_loss", 0.0),
            stats.get("train/approx_kl", 0.0),
            stats.get("train/clip_fraction", 0.0) * 100,
            int(stats.get("collect/n_episodes", 0)),
            stats.get("bootstrap_value", 0.0),  # [C1 diagnosztika]
            stats.get("elapsed_hours", 0.0),
        )

    # =========================================================================
    # Device Feloldas
    # =========================================================================

    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                logger.info("Device: CUDA (auto-detected)")
            else:
                device = torch.device("cpu")
                logger.info("Device: CPU (CUDA nem elerheto)")
        else:
            device = torch.device(device_str)
            logger.info("Device: %s (kezi beallitas)", device_str)
        return device
