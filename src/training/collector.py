"""
Rollout Adatgyujto (collector.py).

Ez a modul a kornyezeti szimulacio futtatasat es a rollout adatok
(trajectories) gyujteset vegzi. A collector a kovetkezo ciklust hajtja vegre:

    1. Megfigyeles kinyerese a kornyezetbol
    2. A halozat forward pass-en keresztul akcio mintavetelezes
    3. A kornyezet leptetes az akcioval
    4. Az atmenet (obs, action, reward, log_prob, value, done) tarolasa

A collector a training/runner.py altal hivott komponens, amely
a buffer.py-be irja az adatokat.

Hivatkozasok:
    - Specifikacio: collector.py — rollout adatok (trajectories) generalasa
    - runner.py: adatgyujtes fazis a event-driven loop-ban
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from src.training.buffer import RolloutBuffer

logger = logging.getLogger(__name__)


# =============================================================================
# Kornyezet Interfesz (Protocol)
# =============================================================================

class PokerEnvironment(Protocol):
    """Protocol interfesz a poker kornyezet szamara.

    Barmely jatekmotor (RLCard, PettingZoo, egyedi) megvalosithatja
    ezt az interfeszt a collector-ral valo kompatibilitas erdekeben.
    
    Megjegyzés: Az RLCard engine step() metódusa 2 értéket ad vissza (obs, reward).
    A done állapot az is_over() metódussal kérdezhetô le.
    """

    def reset(self) -> dict[str, Any]:
        """Uj leosztast indit es visszaadja a kezdo megfigyeles szotarat.

        Returns:
            Nyers jatekallapot szotar (a features.py bemenete).
        """
        ...

    def step(self, action: int) -> tuple[dict[str, Any], float]:
        """Vegrehajt egy akciot es visszaadja az uj allapotot es jutalmat.
        
        RLCard format: 2 értéket ad vissza (obs, reward).
        A done állapot az is_over() metódussal kérdezhetô le.

        Args:
            action: Az akcio indexe (0-8).

        Returns:
            Tuple: (uj_allapot, jutalom)
        """
        ...

    def is_over(self) -> bool:
        """Visszaadja, hogy vége van-e az epizódnak.
        
        Returns:
            True ha az epizód vége, False egyébként.
        """
        ...


# =============================================================================
# Collector Konfiguracio
# =============================================================================

@dataclass(frozen=True)
class CollectorConfig:
    """Az adatgyujto konfiguracioja.

    Attributes:
        rollout_steps: A gyujtendo lepesek szama egy rollout-ban.
        render_interval: Hanyadik epizodonkent logoljon reszletesen (0=soha).
    """

    rollout_steps: int = 2048
    render_interval: int = 0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> CollectorConfig:
        """YAML config szotarbol peldanyosit.

        Args:
            cfg: Teljes YAML konfiguracio.

        Returns:
            CollectorConfig peldany.
        """
        ppo_cfg = cfg.get("ppo", {})
        return cls(
            rollout_steps=ppo_cfg.get("rollout_steps", 2048),
        )


# =============================================================================
# Fo Collector Osztaly
# =============================================================================

class RolloutCollector:
    """A rollout adatgyujtesi ciklus vezerlo osztalya.

    A collector vegigfuttatja a megadott szamu lepest a kornyezetben
    a halozat aktualis policy-jevel, es minden atmenetet a buffer-be ir.

    Example:
        >>> collector = RolloutCollector(config, env, obs_builder, network, buffer)
        >>> stats = collector.collect_rollout()
        >>> print(stats["episodes_completed"])

    Attributes:
        config: Gyujtes konfiguracio.
        env: A poker kornyezet.
        obs_builder: Megfigyeles epito.
        network: Az Actor-Critic halozat.
        buffer: A cel rollout buffer.
    """

    def __init__(
        self,
        config: CollectorConfig,
        env: Any,
        obs_builder: Any,
        network: Any,
        buffer: RolloutBuffer,
        device: torch.device | str = "cpu",
    ) -> None:
        """Inicializalja a collector-t.

        Args:
            config: Gyujtes konfiguracio.
            env: A poker kornyezet (PokerEnvironment protocol).
            obs_builder: ObservationBuilder peldany a features.py-bol.
            network: ActorCriticNetwork peldany.
            buffer: RolloutBuffer peldany.
            device: Szamitasi eszkoz.
        """
        self.config: CollectorConfig = config
        self.env: Any = env
        self.obs_builder: Any = obs_builder
        self.network: Any = network
        self.buffer: RolloutBuffer = buffer
        self.device: torch.device = (
            torch.device(device) if isinstance(device, str) else device
        )

        # Statisztikak
        self._total_steps: int = 0
        self._total_episodes: int = 0
        self._current_obs: dict[str, Any] | None = None

        logger.info(
            "RolloutCollector inicializalva: rollout_steps=%d, device=%s",
            config.rollout_steps, self.device,
        )

    # =========================================================================
    # Fo Gyujtesi Ciklus
    # =========================================================================

    def collect_rollout(self) -> dict[str, float]:
        """Vegrehajt egyetlen rollout adatgyujtest.

        A kovetkezo ciklust hajtja vegre rollout_steps alkalommal:
            1. Ha nincs aktualis megfigyeles -> env.reset()
            2. Megfigyeles epitese (ObservationBuilder)
            3. Halozat forward pass (akcio, log_prob, value)
            4. Kornyezet leptetes (env.step)
            5. Atmenet tarolasa a buffer-be
            6. Ha epizod vege -> reset

        Returns:
            Dict statisztikakkal:
                - steps_collected: Gyujtott lepesek szama
                - episodes_completed: Befejezett epizodok szama
                - mean_reward: Atlagos jutalom per lepes
                - mean_episode_reward: Atlagos jutalom per epizod
                - collection_time_sec: Gyujtes ideje masodpercben
        """
        start_time: float = time.monotonic()
        self.buffer.reset()

        steps_collected: int = 0
        episodes_completed: int = 0
        episode_rewards: list[float] = []
        current_episode_reward: float = 0.0
        step_rewards: list[float] = []

        # Kezdo allapot ha nincs
        if self._current_obs is None:
            self._current_obs = self.env.reset()

        logger.debug(
            "Rollout gyujtes inditas: %d lepes cel",
            self.config.rollout_steps,
        )

        for step_idx in range(self.config.rollout_steps):
            # 1. Megfigyeles epitese
            obs_tensors: dict[str, torch.Tensor] = self.obs_builder.build(
                self._current_obs
            )

            # Batch dimenzio hozzaadasa (1, ...) a halozathoz
            batched_obs: dict[str, torch.Tensor] = {
                k: v.unsqueeze(0).to(self.device) if v.dim() < 3
                else v.unsqueeze(0).to(self.device)
                for k, v in obs_tensors.items()
            }

            # 2. Halozat forward pass (inference mod, no grad)
            with torch.no_grad():
                action, log_prob, entropy, value = self.network.get_action_and_value(
                    batched_obs
                )

            action_int: int = int(action.squeeze().item())
            log_prob_val: torch.Tensor = log_prob.squeeze()
            value_val: torch.Tensor = value.squeeze()

            # 3. Kornyezet leptetes
            # RLCard engine: step() 2 értéket ad vissza (obs, reward)
            # A done állapot az is_over() metódussal kérdezhetô le
            next_obs, reward = self.env.step(action_int)
            done: bool = self.env.is_over()

            # 4. Atmenet tarolasa
            self.buffer.add(
                observation=obs_tensors,
                action=action.squeeze(),
                reward=float(reward),
                log_prob=log_prob_val,
                value=value_val,
                done=done,
            )

            step_rewards.append(float(reward))
            current_episode_reward += float(reward)
            steps_collected += 1
            self._total_steps += 1

            # 5. Epizod vege kezeles
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                episodes_completed += 1
                self._total_episodes += 1
                self._current_obs = self.env.reset()

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "Epizod #%d befejezve: reward=%.2f",
                        self._total_episodes, episode_rewards[-1],
                    )
            else:
                self._current_obs = next_obs

        # Utolso allapot erteke a GAE bootstrap-hoz
        if self._current_obs is not None:
            with torch.no_grad():
                last_obs = self.obs_builder.build(self._current_obs)
                last_batched = {
                    k: v.unsqueeze(0).to(self.device) for k, v in last_obs.items()
                }
                _, _, _, last_value = self.network.get_action_and_value(
                    last_batched
                )
                last_value_float: float = float(last_value.squeeze().item())
        else:
            last_value_float = 0.0

        # GAE szamitas
        self.buffer.compute_gae(last_value=last_value_float)

        # Statisztikak osszeallitasa
        elapsed: float = time.monotonic() - start_time
        mean_reward: float = (
            sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
        )
        mean_ep_reward: float = (
            sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0.0
        )

        stats: dict[str, float] = {
            "steps_collected": float(steps_collected),
            "episodes_completed": float(episodes_completed),
            "mean_step_reward": mean_reward,
            "mean_episode_reward": mean_ep_reward,
            "collection_time_sec": elapsed,
            "steps_per_second": steps_collected / elapsed if elapsed > 0 else 0.0,
            "total_steps": float(self._total_steps),
            "total_episodes": float(self._total_episodes),
        }

        logger.info(
            "Rollout kesz: %d lepes, %d epizod, mean_rew=%.4f, "
            "mean_ep_rew=%.4f, %.1f steps/sec (%.2fs)",
            steps_collected, episodes_completed,
            mean_reward, mean_ep_reward,
            stats["steps_per_second"], elapsed,
        )

        return stats

    # =========================================================================
    # Segedmetodusok
    # =========================================================================

    def get_total_steps(self) -> int:
        """Visszaadja a teljes eddigi lepesszamot.

        Returns:
            Osszesitett lepesszam.
        """
        return self._total_steps

    def get_total_episodes(self) -> int:
        """Visszaadja a teljes eddigi epizodszamot.

        Returns:
            Osszesitett epizodszam.
        """
        return self._total_episodes

    def reset_stats(self) -> None:
        """Nullazza a gyujtesi statisztikakat."""
        self._total_steps = 0
        self._total_episodes = 0
        self._current_obs = None
        logger.debug("Collector statisztikak resetelve.")
