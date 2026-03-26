"""
PPO Trainer (trainer.py).

A Proximal Policy Optimization gradiens frissitesi logikajat
implementalja. Egyetlen iteracion belul a kovetkezo lepeseket hajtja vegre:

    1. A buffer-bol mini-batch-eket kap
    2. Az aktualis halozattal ujra kiertekeli a regi akciokat
    3. Kiszamitja a PPO Clipped Policy Loss-t
    4. Kiszamitja a Value Function Loss-t (opcionalisan clipped)
    5. Kiszamitja az Entropy Bonus-t (felfedezesi osztönzo)
    6. Osszesitett loss alapjan gradiens frissites (Adam)

PPO Loss formula:
    L_CLIP = E[min(r(theta)*A, clip(r(theta), 1-eps, 1+eps)*A)]
    L_VF = 0.5 * E[(V(s) - R)^2]
    L_TOTAL = -L_CLIP + c1*L_VF - c2*H(pi)

Hivatkozasok:
    - Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
    - Specifikacio: trainer.py — PPO gradiens frissites
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.training.buffer import RolloutBuffer

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """A PPO trainer konfiguracios parameterei.

    Attributes:
        learning_rate: Adam optimizer tanulasi rata.
        adam_epsilon: Adam numerikus stabilitas.
        max_grad_norm: Gradiens vagas maximalis normaja.
        clip_epsilon: PPO clipping parameter (epsilon).
        value_loss_coef: Ertek-veszteseg sulyozasi tenyezoje (c1).
        entropy_coef: Entropia regularizacios egyutthato (c2). HOT-RELOADABLE.
        num_epochs: PPO epoch-ok szama egy batch-en belul.
        clip_range_vf: Value function clip range (None=nincs).
        target_kl: Korai leallitas ha a KL divergencia meghaladja.
    """

    learning_rate: float = 3.0e-4
    adam_epsilon: float = 1.0e-5
    max_grad_norm: float = 0.5
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    num_epochs: int = 4
    clip_range_vf: float | None = None
    target_kl: float | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> TrainerConfig:
        """YAML config szotarbol peldanyosit.

        Args:
            cfg: Teljes YAML konfiguracio.

        Returns:
            TrainerConfig peldany.
        """
        ppo = cfg.get("ppo", {})
        return cls(
            learning_rate=ppo.get("learning_rate", 3.0e-4),
            adam_epsilon=ppo.get("adam_epsilon", 1.0e-5),
            max_grad_norm=ppo.get("max_grad_norm", 0.5),
            clip_epsilon=ppo.get("clip_epsilon", 0.2),
            value_loss_coef=ppo.get("value_loss_coefficient", 0.5),
            entropy_coef=ppo.get("entropy_coefficient", 0.01),
            num_epochs=ppo.get("num_epochs", 4),
            clip_range_vf=ppo.get("clip_range_vf"),
            target_kl=None,
        )


class PPOTrainer:
    """Proximal Policy Optimization trainer.

    Kezeli az Adam optimizert, a gradiens vago logikat, es a
    multi-epoch PPO frissitesi ciklust.

    Example:
        >>> trainer = PPOTrainer(config, network)
        >>> stats = trainer.train_on_buffer(buffer)
        >>> print(stats["policy_loss"])

    Attributes:
        config: Trainer konfiguracio.
        network: Az Actor-Critic halozat.
        optimizer: Az Adam optimizer peldany.
    """

    def __init__(
        self,
        config: TrainerConfig,
        network: nn.Module,
        device: torch.device | str = "cpu",
    ) -> None:
        """Inicializalja a PPO trainer-t.

        Args:
            config: Trainer konfiguracio.
            network: ActorCriticNetwork peldany.
            device: Szamitasi eszkoz.
        """
        self.config: TrainerConfig = config
        self.network: nn.Module = network
        self.device: torch.device = (
            torch.device(device) if isinstance(device, str) else device
        )

        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
        )

        self._update_count: int = 0

        logger.info(
            "PPOTrainer inicializalva: lr=%.2e, clip_eps=%.3f, "
            "vf_coef=%.3f, ent_coef=%.4f, epochs=%d, grad_norm=%.2f",
            config.learning_rate, config.clip_epsilon,
            config.value_loss_coef, config.entropy_coef,
            config.num_epochs, config.max_grad_norm,
        )

    # =========================================================================
    # Fo Training Ciklus
    # =========================================================================

    def train_on_buffer(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Vegrehajt egy teljes PPO frissitesi ciklust a buffer adatain.

        A num_epochs alkalommal iteralja vegig a buffer mini-batch-eit,
        es minden batch-re kiszamitja a PPO loss-t es frissiti a sulyokat.

        Args:
            buffer: A feltoltott es GAE-val feldolgozott RolloutBuffer.

        Returns:
            Dict az osszesitett training statisztikakkal:
                - policy_loss: Atlagos policy loss
                - value_loss: Atlagos value loss
                - entropy_loss: Atlagos entropy bonus
                - total_loss: Atlagos osszesitett loss
                - approx_kl: Becsult KL divergencia
                - clip_fraction: A vágott rátaar aranya
                - explained_variance: Megmagyarazott variancia
                - grad_norm: Atlagos gradiens norma
                - updates: Frissitesek szama
                - train_time_sec: Training ido masodpercben
        """
        start_time: float = time.monotonic()
        self.network.train()

        # Aggregalt metrikak
        total_policy_loss: float = 0.0
        total_value_loss: float = 0.0
        total_entropy: float = 0.0
        total_loss_sum: float = 0.0
        total_approx_kl: float = 0.0
        total_clip_frac: float = 0.0
        total_grad_norm: float = 0.0
        num_updates: int = 0
        early_stop: bool = False

        for epoch in range(self.config.num_epochs):
            if early_stop:
                break

            for batch in buffer.get_mini_batches():
                loss_info = self._compute_and_step(batch)

                total_policy_loss += loss_info["policy_loss"]
                total_value_loss += loss_info["value_loss"]
                total_entropy += loss_info["entropy"]
                total_loss_sum += loss_info["total_loss"]
                total_approx_kl += loss_info["approx_kl"]
                total_clip_frac += loss_info["clip_fraction"]
                total_grad_norm += loss_info["grad_norm"]
                num_updates += 1

                # Korai leallitas KL divergencia alapjan
                if (self.config.target_kl is not None
                        and loss_info["approx_kl"] > 1.5 * self.config.target_kl):
                    logger.warning(
                        "Korai leallitas: KL=%.4f > %.4f (target*1.5), "
                        "epoch=%d/%d",
                        loss_info["approx_kl"],
                        1.5 * self.config.target_kl,
                        epoch + 1, self.config.num_epochs,
                    )
                    early_stop = True
                    break

            logger.debug(
                "PPO Epoch %d/%d kesz: pl=%.4f, vl=%.4f, H=%.4f, kl=%.4f",
                epoch + 1, self.config.num_epochs,
                total_policy_loss / max(num_updates, 1),
                total_value_loss / max(num_updates, 1),
                total_entropy / max(num_updates, 1),
                total_approx_kl / max(num_updates, 1),
            )

        self._update_count += num_updates
        elapsed: float = time.monotonic() - start_time
        n: int = max(num_updates, 1)

        stats: dict[str, float] = {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy_loss": total_entropy / n,
            "total_loss": total_loss_sum / n,
            "approx_kl": total_approx_kl / n,
            "clip_fraction": total_clip_frac / n,
            "grad_norm": total_grad_norm / n,
            "updates": float(num_updates),
            "train_time_sec": elapsed,
            "total_updates": float(self._update_count),
        }

        logger.info(
            "PPO frissites kesz: %d update, pl=%.4f, vl=%.4f, "
            "H=%.4f, kl=%.4f, clip=%.2f%%, grad=%.3f (%.2fs)",
            num_updates,
            stats["policy_loss"], stats["value_loss"],
            stats["entropy_loss"], stats["approx_kl"],
            stats["clip_fraction"] * 100, stats["grad_norm"],
            elapsed,
        )

        return stats

    # =========================================================================
    # Egy Mini-Batch Feldolgozasa
    # =========================================================================

    def _compute_and_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Egy mini-batch-re kiszamitja a loss-t es vegrehajtja a gradiens lepest.

        Args:
            batch: A buffer.get_mini_batches() altal szolgaltatott szotar.

        Returns:
            Dict az egyes loss komponensekkel.
        """
        # Adatok device-ra mozgatasa
        observations: dict[str, torch.Tensor] = {
            k: v.to(self.device) for k, v in batch["observations"].items()
        }
        actions: torch.Tensor = batch["actions"].to(self.device)
        old_log_probs: torch.Tensor = batch["old_log_probs"].to(self.device)
        advantages: torch.Tensor = batch["advantages"].to(self.device)
        returns: torch.Tensor = batch["returns"].to(self.device)
        old_values: torch.Tensor = batch["old_values"].to(self.device)

        # Uj kiertekeles az aktualis halozattal
        _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
            observations, action=actions.long()
        )
        new_values = new_values.squeeze(-1)

        # === PPO Clipped Policy Loss ===
        log_ratio: torch.Tensor = new_log_probs - old_log_probs
        ratio: torch.Tensor = torch.exp(log_ratio)

        # Clipped surrogate
        surr1: torch.Tensor = ratio * advantages
        surr2: torch.Tensor = (
            torch.clamp(ratio, 1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon)
            * advantages
        )
        policy_loss: torch.Tensor = -torch.min(surr1, surr2).mean()

        # === Value Function Loss ===
        if self.config.clip_range_vf is not None:
            # Clipped value loss
            v_clipped: torch.Tensor = old_values + torch.clamp(
                new_values - old_values,
                -self.config.clip_range_vf,
                self.config.clip_range_vf,
            )
            vf_loss1: torch.Tensor = (new_values - returns) ** 2
            vf_loss2: torch.Tensor = (v_clipped - returns) ** 2
            value_loss: torch.Tensor = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
        else:
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()

        # === Entropy Bonus ===
        entropy_loss: torch.Tensor = entropy.mean()

        # === Osszesitett Loss ===
        total_loss: torch.Tensor = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * entropy_loss
        )

        # === Gradiens Frissites ===
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradiens norma szamitas es vagas
        grad_norm: float = float(
            nn.utils.clip_grad_norm_(
                self.network.parameters(), self.config.max_grad_norm
            ).item()
        )

        self.optimizer.step()

        # === Diagnosztika ===
        with torch.no_grad():
            approx_kl: float = float(((ratio - 1) - log_ratio).mean().item())
            clip_fraction: float = float(
                ((ratio - 1.0).abs() > self.config.clip_epsilon)
                .float().mean().item()
            )

        # NaN/Inf ellenorzes
        if not torch.isfinite(total_loss):
            logger.error(
                "KRITIKUS: NaN/Inf loss detektalva! "
                "pl=%.4f, vl=%.4f, H=%.4f, total=%.4f",
                policy_loss.item(), value_loss.item(),
                entropy_loss.item(), total_loss.item(),
            )

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy_loss.item()),
            "total_loss": float(total_loss.item()),
            "approx_kl": approx_kl,
            "clip_fraction": clip_fraction,
            "grad_norm": grad_norm,
        }

    # =========================================================================
    # Hiperparameter Hot-Reload
    # =========================================================================

    def update_learning_rate(self, new_lr: float) -> None:
        """Futasideju tanulasi rata modositas (hot-reload).

        Args:
            new_lr: Az uj tanulasi rata.
        """
        old_lr: float = self.optimizer.param_groups[0]["lr"]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        self.config.learning_rate = new_lr
        logger.info("Tanulasi rata frissitve: %.2e -> %.2e", old_lr, new_lr)

    def update_entropy_coef(self, new_coef: float) -> None:
        """Futasideju entropia egyutthato modositas (hot-reload).

        Az Orchestrator hivja meg stagnacio vagy passzivitas eszlelesekor.

        Args:
            new_coef: Az uj entropia koefficians.
        """
        old_coef: float = self.config.entropy_coef
        self.config.entropy_coef = new_coef
        logger.info(
            "Entropia koefficians frissitve: %.4f -> %.4f",
            old_coef, new_coef,
        )

    def update_clip_epsilon(self, new_eps: float) -> None:
        """Futasideju clip epsilon modositas.

        Args:
            new_eps: Az uj PPO clip epsilon.
        """
        old_eps: float = self.config.clip_epsilon
        self.config.clip_epsilon = new_eps
        logger.info("Clip epsilon frissitve: %.3f -> %.3f", old_eps, new_eps)

    def get_optimizer_state(self) -> dict[str, Any]:
        """Visszaadja az optimizer allapotot checkpoint menteshez.

        Returns:
            Az optimizer state_dict().
        """
        return self.optimizer.state_dict()

    def load_optimizer_state(self, state_dict: dict[str, Any]) -> None:
        """Visszaallitja az optimizer allapotot checkpoint betoltesbol.

        Args:
            state_dict: Az optimizer korabbi state_dict()-je.
        """
        self.optimizer.load_state_dict(state_dict)
        logger.info("Optimizer allapot visszaallitva.")
