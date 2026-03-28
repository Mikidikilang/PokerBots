"""
Rollout Buffer a PPO Tapasztalatok Tarolasahoz (buffer.py).

Ez a modul a PPO on-policy algoritmus rollout adatait (trajectories)
tarolja es kezeli. A buffer egy epizodon belul gyujtott atmenetek
(state, action, reward, log_prob, value, done) szekvencialis tarolojat,
a Generalized Advantage Estimation (GAE) szamitasat, es a mini-batch
mintavetelezes logikat implementalja.

Produkcios kornyezetben a TorchRL CompressedListStorage + Zstandard
tomoritest hasznalja; ha a torchrl nem elerheto, egy egyszeru
in-memory fallback lep eletbe.

Hivatkozasok:
    - Specifikacio: training/buffer.py — CompressedListStorage
    - GAE: Schulman et al. "High-Dimensional Continuous Control Using GAE"
    - TorchRL: https://docs.pytorch.org/rl/stable/

FIX C1 (2025-03-28): set_last_value() / get_last_bootstrap_value() hozzaadva.
    A korabbi implementacioban a buffer nem tarolta a bootstrap erteket,
    ezert a runner.py a collector._current_obs-bol szamolta ujra, ami
    episode-hataron versenyfutasi allapotot (race condition) okozott.
    Most a collector.collect_rollout() vege elott atomikusan beallitja
    a bootstrap erteket a bufferben, es a runner.py onnan olvassa ki.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Generator

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class RolloutBufferConfig:
    """A rollout buffer konfiguracios parameterei.

    Attributes:
        buffer_size: Maximalis tarolhato lepesek szama.
        gamma: Diszkont faktor a jutalom szamitashoz.
        gae_lambda: GAE lambda parameter.
        num_mini_batches: Mini-batch-ek szama a PPO epoch-okhoz.
        compression_enabled: Zstandard tomoritest hasznaljon-e.
        compression_level: Zstd tomorites szint (1-22).
    """

    buffer_size: int = 2048
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_mini_batches: int = 4
    compression_enabled: bool = True
    compression_level: int = 3

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> RolloutBufferConfig:
        """YAML config szotarbol peldanyosit.

        Args:
            cfg: Teljes YAML konfiguracio.

        Returns:
            RolloutBufferConfig peldany.
        """
        ppo_cfg = cfg.get("ppo", {})
        buf_cfg = ppo_cfg.get("buffer", {})
        return cls(
            buffer_size=ppo_cfg.get("rollout_steps", 2048),
            gamma=ppo_cfg.get("gamma", 0.99),
            gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
            num_mini_batches=ppo_cfg.get("num_mini_batches", 4),
            compression_enabled=buf_cfg.get("compression", "zstandard") == "zstandard",
            compression_level=buf_cfg.get("compression_level", 3),
        )


class RolloutBuffer:
    """On-policy rollout buffer a PPO algoritmushoz.

    A buffer szekvencialisan gyujti a rollout adatokat, majd a
    compute_gae() hivassal kiszamitja az advantage ertekeket es
    a diszkontalt return-oket. Ezutan a get_mini_batches()
    generatorral veletlenszeruen feldarabolt mini-batch-eket
    szolgaltat a PPO epoch-ok szamara.

    [C1 FIX] A bootstrap erteket a collector atomikusan tarolja a
    bufferben a collect_rollout() legvegen, set_last_value() segitsegevel.
    A runner.py a compute_gae(last_value=self.buffer.get_last_bootstrap_value())
    hivason keresztul olvassa ki. Ez megszunteti a korabbi race conditiont,
    ahol a runner.py a collector._current_obs-bol szamolta ujra az erteket
    egy lepessessel kesobb.

    Example:
        >>> buf = RolloutBuffer(RolloutBufferConfig(buffer_size=2048))
        >>> for step in range(2048):
        ...     buf.add(obs, action, reward, log_prob, value, done)
        >>> # A collector automatikusan beallitja:
        >>> buf.set_last_value(bootstrap_val)
        >>> buf.compute_gae(last_value=buf.get_last_bootstrap_value())
        >>> for batch in buf.get_mini_batches():
        ...     loss = trainer.compute_loss(batch)

    Attributes:
        config: A buffer konfiguracioja.
        pos: Az aktualis irasi pozicio.
        full: True ha a buffer megtelt.
    """

    def __init__(self, config: RolloutBufferConfig | None = None) -> None:
        """Inicializalja a rollout buffert.

        Args:
            config: Buffer konfiguracio. Alapertelmezett ha None.
        """
        self.config: RolloutBufferConfig = config or RolloutBufferConfig()
        self.pos: int = 0
        self.full: bool = False

        # Tarolo tombokre: listakent indul, a compute_gae-ben tensorra konvertalodik
        self._observations: list[dict[str, torch.Tensor]] = []
        self._actions: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self._log_probs: list[torch.Tensor] = []
        self._values: list[torch.Tensor] = []
        self._dones: list[bool] = []

        # GAE szamitas eredmenyei (compute_gae() tolti fel)
        self._advantages: torch.Tensor | None = None
        self._returns: torch.Tensor | None = None

        # [C1 FIX] Atomikus bootstrap ertek taroloja.
        # A collector.collect_rollout() a rollout VEGE elott beallitja,
        # mielott a runner.py meghivja compute_gae()-t.
        # Ez garantalja, hogy a V(s_T) a HELYES, truncated allapothoz tartozik.
        self._last_bootstrap_value: float = 0.0

        # Pre-consolidated batching tensors (allocated in compute_gae)
        self._obs_tensors: dict[str, torch.Tensor] = {}
        self._actions_tensor: torch.Tensor | None = None
        self._log_probs_tensor: torch.Tensor | None = None
        self._values_tensor: torch.Tensor | None = None

        logger.info(
            "RolloutBuffer inicializalva: size=%d, gamma=%.3f, "
            "gae_lambda=%.3f, mini_batches=%d",
            self.config.buffer_size, self.config.gamma, self.config.gae_lambda,
            self.config.num_mini_batches,
        )

    # =========================================================================
    # [C1 FIX] Bootstrap Ertek Kezelese
    # =========================================================================

    def set_last_value(self, value: float | torch.Tensor) -> None:
        """Beallitja a GAE szamitashoz szukseges bootstrap erteket.

        Ezt a metodust a collector.collect_rollout() hivja meg kozvetlenul
        a rollout loop vege utan, MIELOTT visszaadna a vezerlesst a runner-nek.
        Igy a bootstrap ertek garantaltan a HELYES, truncated allapothoz
        tartozo V(s_T), nem pedig egy kesobb szamolt kozelites.

        [C1 FIX] A korabbi implementacioban a RolloutBuffer.set_last_value()
        nem letezett, ezert a collector `if hasattr(self.buffer, "set_last_value")`
        ag sohasem futott le. A runner.py ezutan ujra hivta
        collector.get_last_bootstrap_value()-t, ami a mar megvaltoztatott
        _current_obs-bol szamolt - igy a bootstrap ertek egy lepessessel
        kesobb szamolodott, versenyfutasi allapotot eloallitva episode hatarokon.

        Args:
            value: Bootstrap ertek V(s_T). Lehet float vagy torch.Tensor.
                   Tensor eseten .detach().cpu().item() hivodik ra.
        """
        if isinstance(value, torch.Tensor):
            self._last_bootstrap_value = float(value.detach().cpu().item())
        else:
            self._last_bootstrap_value = float(value)
        logger.debug("Bootstrap ertek beallitva: %.6f", self._last_bootstrap_value)

    def get_last_bootstrap_value(self) -> float:
        """Visszaadja a set_last_value() altal beallitott bootstrap erteket.

        Ha set_last_value() meg nem lett meghivva (pl. az epizod normalis
        vegfutasa volt, done=True), visszaad 0.0-t, ami matematikailag
        helyes (nincs jovobeli jutalom terminalis allapotbol).

        Returns:
            A legutolso bootstrap ertek, alapertelmezetten 0.0.
        """
        return self._last_bootstrap_value

    # =========================================================================
    # Adat Hozzaadas
    # =========================================================================

    def add(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor,
        reward: float,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        done: bool,
    ) -> None:
        """Egyetlen atmenet (transition) hozzaadasa a bufferhez.

        Args:
            observation: Megfigyeles szotar (a features.py kimenete).
            action: A kivalasztott akcio indexe.
            reward: A kornyezet altal adott jutalom.
            log_prob: Az akcio log-valoszinusege pi(a|s).
            value: Az allapot erteke V(s) a Critic-bol.
            done: True ha az epizod veget ert.
        """
        self._observations.append(observation)
        self._actions.append(action.detach())
        self._rewards.append(reward)
        self._log_probs.append(log_prob.detach())
        self._values.append(value.detach())
        self._dones.append(done)
        self.pos += 1

        if self.pos >= self.config.buffer_size:
            self.full = True

        if self.pos % 500 == 0:
            logger.debug("Buffer pos: %d/%d", self.pos, self.config.buffer_size)

    # =========================================================================
    # GAE Szamitas
    # =========================================================================

    def compute_gae(self, last_value: float = 0.0) -> None:
        """Kiszamitja a Generalized Advantage Estimation (GAE) ertekeket.

        A GAE formulaja:
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}

        A returns a diszkontalt jutalom: R_t = A_t + V(s_t)

        [C1 FIX] Ajanlott hivasi mod (runner.py-ban):
            self.buffer.compute_gae(last_value=self.buffer.get_last_bootstrap_value())
        A last_value parameter megtartva a backward kompatibilitashoz, de a
        set_last_value() + get_last_bootstrap_value() paron keresztul torteno
        hivatas garantaltan helyes timing-ot biztosit.

        Args:
            last_value: Az utolso allapot erteke V(s_T) a bootstrap-hoz.
                       0.0 ha az epizod veget ert. Elonyben reszesitett mod:
                       self.buffer.get_last_bootstrap_value() hasznalata.
        """
        num_steps: int = len(self._rewards)
        if num_steps == 0:
            logger.warning("compute_gae: ures buffer, nincs mit szamolni.")
            return

        advantages: np.ndarray = np.zeros(num_steps, dtype=np.float32)
        values_np: np.ndarray = np.array(
            [v.item() if isinstance(v, torch.Tensor) else float(v) for v in self._values],
            dtype=np.float32,
        )
        rewards_np: np.ndarray = np.array(self._rewards, dtype=np.float32)
        dones_np: np.ndarray = np.array(self._dones, dtype=np.float32)

        gamma: float = self.config.gamma
        lam: float = self.config.gae_lambda

        last_gae: float = 0.0
        next_value: float = last_value

        # Visszafele iteralas (reverse sweep)
        for t in reversed(range(num_steps)):
            non_terminal: float = 1.0 - dones_np[t]
            delta: float = (
                rewards_np[t]
                + gamma * next_value * non_terminal
                - values_np[t]
            )
            last_gae = delta + gamma * lam * non_terminal * last_gae
            advantages[t] = last_gae
            next_value = values_np[t]

        self._advantages = torch.tensor(advantages, dtype=torch.float32)
        self._returns = self._advantages + torch.tensor(values_np, dtype=torch.float32)

        # Advantage normalizalas (zero-mean, unit-variance)
        adv_mean: float = float(self._advantages.mean().item())
        adv_std: float = float(self._advantages.std().item()) + 1e-8
        self._advantages = (self._advantages - adv_mean) / adv_std

        logger.info(
            "GAE szamitas kesz: %d lepes, adv_mean=%.4f, adv_std=%.4f, "
            "returns_mean=%.4f, bootstrap_val=%.6f",
            num_steps, adv_mean, adv_std,
            float(self._returns.mean().item()),
            last_value,
        )

        # Pre-allocate consolidated batching tensors (O(1) per-batch indexing)
        self._consolidate_tensors()

    def _consolidate_tensors(self) -> None:
        """Pre-allocate consolidated tensors for O(1) mini-batch building."""
        num_steps: int = len(self._rewards)

        if self._observations:
            first_obs = self._observations[0]
            self._obs_tensors = {}
            for key in first_obs:
                self._obs_tensors[key] = torch.stack(
                    [self._observations[i][key] for i in range(num_steps)],
                    dim=0
                )
        else:
            self._obs_tensors = {}

        self._actions_tensor = torch.stack(self._actions, dim=0).view(num_steps)
        self._log_probs_tensor = torch.stack(self._log_probs, dim=0).view(num_steps)
        values_stacked: torch.Tensor = torch.stack(
            [v if isinstance(v, torch.Tensor) else torch.tensor(float(v))
             for v in self._values], dim=0
        )
        self._values_tensor = values_stacked.view(num_steps)

        logger.debug(
            "Consolidated batching tensors allocated: %d steps, obs keys=%d",
            num_steps, len(self._obs_tensors),
        )

    # =========================================================================
    # Mini-Batch Mintavetelezes
    # =========================================================================

    def get_mini_batches(self) -> Generator[dict[str, Any], None, None]:
        """Veletlenszeruen kevert mini-batch-eket general a PPO epoch-okhoz.

        Yields:
            Dict[str, Any] mini-batch szotar.

        Raises:
            RuntimeError: Ha a compute_gae() meg nem lett meghivva.
        """
        if self._advantages is None or self._returns is None:
            raise RuntimeError(
                "A compute_gae() meg nem lett meghivva. "
                "Hivd meg a get_mini_batches() elott."
            )

        if not self._obs_tensors:
            raise RuntimeError(
                "Consolidated tensors not populated. "
                "This indicates compute_gae() was not called after the last reset()."
            )

        num_steps: int = len(self._rewards)
        batch_size: int = num_steps // self.config.num_mini_batches

        if batch_size < 1:
            logger.warning(
                "Mini-batch meret < 1: %d lepes / %d batch. "
                "Egyetlen batch-kent szolgaltatas.",
                num_steps, self.config.num_mini_batches,
            )
            batch_size = num_steps

        indices: np.ndarray = np.random.permutation(num_steps)

        for start in range(0, num_steps, batch_size):
            end: int = min(start + batch_size, num_steps)
            batch_indices: np.ndarray = indices[start:end]

            batch_obs: dict[str, torch.Tensor] = {
                key: obs_tensor[batch_indices]
                for key, obs_tensor in self._obs_tensors.items()
            }

            batch: dict[str, Any] = {
                "observations": batch_obs,
                "actions": self._actions_tensor[batch_indices],
                "old_log_probs": self._log_probs_tensor[batch_indices],
                "advantages": self._advantages[batch_indices],
                "returns": self._returns[batch_indices],
                "old_values": self._values_tensor[batch_indices],
            }

            logger.debug(
                "Mini-batch: %d mintak (indices %d-%d)",
                len(batch_indices), start, end,
            )
            yield batch

    # =========================================================================
    # Buffer Kezeles
    # =========================================================================

    def reset(self) -> None:
        """Torli a buffer teljes tartalmat a kovetkezo rollout-hoz.

        [C1 FIX] A _last_bootstrap_value is visszaall 0.0-ra, hogy ne
        maradjon stale ertek a kovetkezo rollout GAE szamitasahoz.
        """
        self._observations.clear()
        self._actions.clear()
        self._rewards.clear()
        self._log_probs.clear()
        self._values.clear()
        self._dones.clear()
        self._advantages = None
        self._returns = None
        self.pos = 0
        self.full = False

        # [C1 FIX] Bootstrap ertek resetelese — stale ertek ne maradjon
        self._last_bootstrap_value = 0.0

        # Explicitly release consolidated tensors
        self._obs_tensors = {}
        self._actions_tensor = None
        self._log_probs_tensor = None
        self._values_tensor = None
        logger.debug("RolloutBuffer resetelve (bootstrap ertek es konszolidalt tenzorok torolve).")

    def __len__(self) -> int:
        """A bufferben tarolt lepesek szama."""
        return len(self._rewards)

    def get_stats(self) -> dict[str, float]:
        """Visszaadja a buffer tartalmanak osszefoglalo statisztikait."""
        stats: dict[str, float] = {
            "buffer_size": float(len(self)),
            "buffer_full": float(self.full),
            "last_bootstrap_value": self._last_bootstrap_value,
        }
        if self._rewards:
            rewards = np.array(self._rewards)
            stats["reward_mean"] = float(rewards.mean())
            stats["reward_std"] = float(rewards.std())
            stats["reward_min"] = float(rewards.min())
            stats["reward_max"] = float(rewards.max())
            stats["episode_dones"] = float(sum(self._dones))
        if self._advantages is not None:
            stats["advantage_mean"] = float(self._advantages.mean().item())
            stats["advantage_std"] = float(self._advantages.std().item())
        if self._returns is not None:
            stats["returns_mean"] = float(self._returns.mean().item())
        return stats

    def save_to_disk(self, filepath: str) -> None:
        """A buffer tartalmat diszkre menti (checkpoint szamara)."""
        state: dict[str, Any] = {
            "rewards": self._rewards,
            "dones": self._dones,
            "pos": self.pos,
            "full": self.full,
            "config": self.config,
            "last_bootstrap_value": self._last_bootstrap_value,  # [C1 FIX]
        }
        torch.save(state, filepath)
        logger.info("Buffer state mentve: %s (%d lepes)", filepath, len(self))

    def load_from_disk(self, filepath: str) -> None:
        """A buffer tartalmat visszatolti a diszkrol."""
        state: dict[str, Any] = torch.load(filepath, weights_only=True)
        self._rewards = state["rewards"]
        self._dones = state["dones"]
        self.pos = state["pos"]
        self.full = state["full"]
        # [C1 FIX] Bootstrap ertek visszatoltes
        self._last_bootstrap_value = state.get("last_bootstrap_value", 0.0)
        logger.info("Buffer state betoltve: %s (%d lepes)", filepath, self.pos)
