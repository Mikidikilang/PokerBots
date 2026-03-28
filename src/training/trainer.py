"""
PPO Trainer (trainer.py).

[FIX C-4 — 2025-03-28] Cross-rank NaN guard added to _compute_and_step().

    ROOT CAUSE:
    If rank 0 detects a NaN/Inf loss and raises FloatingPointError, it exits
    the training loop. Rank 1 is still inside loss.backward() waiting for
    rank 0's all_reduce contribution — it hangs permanently. The entire
    2xT4 session locks up without any error message.

    THE FIX:
    After computing total_loss (before backward), perform a dist.all_reduce()
    with ReduceOp.MAX on a NaN flag tensor. This is a synchronous collective:
    ALL ranks either raise FloatingPointError together (if any rank has NaN)
    or ALL proceed to backward() together. No asymmetric exits possible.
    In single-GPU mode (world_size=1 or dist not initialized) this is a no-op.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.training.buffer import RolloutBuffer

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """A PPO trainer konfiguracios parameterei."""

    learning_rate: float = 3.0e-4
    adam_epsilon: float = 1.0e-5
    max_grad_norm: float = 0.5
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    num_epochs: int = 4
    clip_range_vf: float | None = None
    target_kl: float | None = 0.015   # [Priority-2] default KL threshold; config.yaml overrides
    learning_rate_schedule: str = "none"
    lr_warmup_steps: int | None = None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> TrainerConfig:
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
            target_kl=ppo.get("target_kl", 0.015),  # [Priority-2] read from config
            learning_rate_schedule=ppo.get("learning_rate_schedule", "none"),
            lr_warmup_steps=ppo.get("lr_warmup_steps"),
        )


class PPOTrainer:
    """Proximal Policy Optimization trainer."""

    def __init__(
        self,
        config: TrainerConfig,
        network: nn.Module,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config: TrainerConfig = config
        self.network: nn.Module = network
        self.device: torch.device = (
            torch.device(device) if isinstance(device, str) else device
        )
        self.network = self.network.to(self.device)

        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            self.network.parameters(),
            lr=config.learning_rate,
            eps=config.adam_epsilon,
        )

        self.scheduler: torch.optim.lr_scheduler.LambdaLR | None = None
        if config.learning_rate_schedule == "linear":
            def lr_lambda(step: int) -> float:
                warmup_steps = config.lr_warmup_steps or 0
                if step < warmup_steps:
                    return float(step) / max(1, warmup_steps)
                return 1.0
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda
            )

        self._update_count: int = 0

        logger.info(
            "PPOTrainer inicializalva: lr=%.2e, clip_eps=%.3f, "
            "vf_coef=%.3f, ent_coef=%.4f, epochs=%d, grad_norm=%.2f, device=%s",
            config.learning_rate, config.clip_epsilon,
            config.value_loss_coef, config.entropy_coef,
            config.num_epochs, config.max_grad_norm, self.device,
        )

    # =========================================================================
    # Training Loop
    # =========================================================================

    def train_on_buffer(self, buffer: RolloutBuffer) -> dict[str, float]:
        start_time: float = time.monotonic()
        self.network.train()

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
                total_value_loss  += loss_info["value_loss"]
                total_entropy     += loss_info["entropy"]
                total_loss_sum    += loss_info["total_loss"]
                total_approx_kl   += loss_info["approx_kl"]
                total_clip_frac   += loss_info["clip_fraction"]
                total_grad_norm   += loss_info["grad_norm"]
                num_updates += 1

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
                total_value_loss  / max(num_updates, 1),
                total_entropy     / max(num_updates, 1),
                total_approx_kl   / max(num_updates, 1),
            )

        self._update_count += num_updates
        elapsed: float = time.monotonic() - start_time
        n: int = max(num_updates, 1)

        stats: dict[str, float] = {
            "policy_loss":    total_policy_loss / n,
            "value_loss":     total_value_loss  / n,
            "entropy_loss":   total_entropy     / n,
            "total_loss":     total_loss_sum    / n,
            "approx_kl":      total_approx_kl   / n,
            "clip_fraction":  total_clip_frac   / n,
            "grad_norm":      total_grad_norm   / n,
            "updates":        float(num_updates),
            "train_time_sec": elapsed,
            "total_updates":  float(self._update_count),
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
    # Mini-Batch Step
    # =========================================================================

    def _compute_and_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Compute PPO loss and execute one optimizer step.

        [FIX C-4] Cross-rank NaN guard via dist.all_reduce() before backward().

        A NaN on any single rank is broadcast to ALL ranks, so all ranks
        raise FloatingPointError together. This prevents the scenario where
        rank 0 raises and exits while rank 1 blocks in all_reduce inside
        backward() — which previously caused a permanent hang.

        In single-GPU mode (dist not initialized) the guard is a no-op.
        """
        observations: dict[str, torch.Tensor] = {
            k: v.to(self.device) for k, v in batch["observations"].items()
        }
        actions:        torch.Tensor = batch["actions"].to(self.device)
        old_log_probs:  torch.Tensor = batch["old_log_probs"].to(self.device)
        advantages:     torch.Tensor = batch["advantages"].to(self.device)
        returns:        torch.Tensor = batch["returns"].to(self.device)
        old_values:     torch.Tensor = batch["old_values"].to(self.device)

        action_dist, new_values = self.network(observations)
        new_log_probs = action_dist.log_prob(actions.long())
        entropy       = action_dist.entropy()
        new_values    = new_values.squeeze(-1)

        # === PPO Clipped Policy Loss ===
        log_ratio: torch.Tensor = new_log_probs - old_log_probs
        ratio:     torch.Tensor = torch.exp(log_ratio)

        surr1: torch.Tensor = ratio * advantages
        surr2: torch.Tensor = (
            torch.clamp(ratio, 1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon)
            * advantages
        )
        policy_loss: torch.Tensor = -torch.min(surr1, surr2).mean()

        # === Value Function Loss ===
        if self.config.clip_range_vf is not None:
            v_clipped: torch.Tensor = old_values + torch.clamp(
                new_values - old_values,
                -self.config.clip_range_vf,
                self.config.clip_range_vf,
            )
            vf_loss1: torch.Tensor = (new_values - returns) ** 2
            vf_loss2: torch.Tensor = (v_clipped  - returns) ** 2
            value_loss: torch.Tensor = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
        else:
            value_loss = 0.5 * ((new_values - returns) ** 2).mean()

        # === Entropy Bonus ===
        entropy_loss: torch.Tensor = entropy.mean()

        # === Total Loss ===
        total_loss: torch.Tensor = (
            policy_loss
            + self.config.value_loss_coef * value_loss
            - self.config.entropy_coef * entropy_loss
        )

        # ===================================================================
        # [FIX C-4] Cross-rank NaN guard — MUST run before optimizer.zero_grad()
        # and before backward(). All ranks participate in the all_reduce so
        # either all raise or all proceed — no asymmetric exits possible.
        # ===================================================================
        self._cross_rank_nan_guard(
            total_loss, policy_loss, value_loss, entropy_loss
        )
        # ===================================================================

        # === Local NaN check (single-GPU or after cross-rank guard) ===
        if not torch.isfinite(total_loss):
            logger.error(
                "KRITIKUS: NaN/Inf loss! pl=%.4f, vl=%.4f, H=%.4f, total=%.4f",
                policy_loss.item(), value_loss.item(),
                entropy_loss.item(), total_loss.item(),
            )
            raise FloatingPointError(
                f"NaN/Inf loss: policy={policy_loss.item()}, "
                f"value={value_loss.item()}, total={total_loss.item()}"
            )

        # === Gradient step ===
        self.optimizer.zero_grad()
        total_loss.backward()

        for name, param in self.network.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                nan_count = (~torch.isfinite(param.grad)).sum().item()
                logger.critical(
                    "Gradient NaN in %s: %d/%d elements — aborting step",
                    name, nan_count, param.grad.numel(),
                )
                raise FloatingPointError(
                    f"NaN/Inf gradient in {name}: {nan_count} elements."
                )

        grad_norm: float = float(
            nn.utils.clip_grad_norm_(
                self.network.parameters(), self.config.max_grad_norm
            ).item()
        )

        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        with torch.inference_mode():
            approx_kl:    float = float(((ratio - 1) - log_ratio).mean().item())
            clip_fraction: float = float(
                ((ratio - 1.0).abs() > self.config.clip_epsilon)
                .float().mean().item()
            )

        return {
            "policy_loss":  float(policy_loss.item()),
            "value_loss":   float(value_loss.item()),
            "entropy":      float(entropy_loss.item()),
            "total_loss":   float(total_loss.item()),
            "approx_kl":    approx_kl,
            "clip_fraction": clip_fraction,
            "grad_norm":    grad_norm,
        }

    # =========================================================================
    # [FIX C-4] Cross-rank NaN synchronization helper
    # =========================================================================

    @staticmethod
    def _cross_rank_nan_guard(
        total_loss:  torch.Tensor,
        policy_loss: torch.Tensor,
        value_loss:  torch.Tensor,
        entropy_loss: torch.Tensor,
    ) -> None:
        """All-reduce a NaN flag so all DDP ranks raise together.

        In single-GPU mode this is a complete no-op (dist not initialized).
        In multi-GPU DDP mode this ensures symmetric exit: if ANY rank has
        a NaN/Inf loss, ALL ranks receive that information and ALL raise
        FloatingPointError before reaching loss.backward().

        Without this guard:
            rank 0 raises FloatingPointError → exits training loop
            rank 1 enters next iteration → calls loss.backward() → waits for
            rank 0's all_reduce contribution → hangs permanently.

        With this guard:
            all ranks call all_reduce(MAX) → any NaN is visible to all →
            all ranks raise → DDP process group destroyed cleanly.

        Args:
            total_loss:   Scalar loss tensor (used for NaN detection).
            policy_loss:  For diagnostic message only.
            value_loss:   For diagnostic message only.
            entropy_loss: For diagnostic message only.
        """
        try:
            import torch.distributed as dist
            if not (dist.is_available() and dist.is_initialized()):
                return  # Single-GPU — no-op
        except ImportError:
            return

        is_nan_local = 0.0 if torch.isfinite(total_loss) else 1.0
        nan_flag = torch.tensor(
            [is_nan_local],
            dtype=torch.float32,
            device=total_loss.device,
        )

        # Synchronous collective: blocks until all ranks contribute.
        # MAX ensures any rank's NaN propagates to all ranks.
        dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)

        if nan_flag.item() > 0:
            rank = dist.get_rank()
            logger.critical(
                "[Rank %d] Cross-rank NaN guard triggered: "
                "at least one DDP rank has NaN/Inf loss. "
                "local: policy=%.4f, value=%.4f, entropy=%.4f, total=%.4f. "
                "All ranks will raise together to prevent backward() hang.",
                rank,
                policy_loss.item(), value_loss.item(),
                entropy_loss.item(), total_loss.item(),
            )
            raise FloatingPointError(
                f"[Rank {rank}] NaN/Inf detected on at least one DDP rank. "
                f"Local: policy={policy_loss.item():.4f}, "
                f"value={value_loss.item():.4f}, total={total_loss.item():.4f}"
            )

    # =========================================================================
    # Hot-Reload
    # =========================================================================

    def update_learning_rate(self, new_lr: float) -> None:
        old_lr: float = self.optimizer.param_groups[0]["lr"]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        self.config.learning_rate = new_lr
        logger.info("Tanulasi rata frissitve: %.2e -> %.2e", old_lr, new_lr)

    def update_entropy_coef(self, new_coef: float) -> None:
        old_coef: float = self.config.entropy_coef
        self.config.entropy_coef = new_coef
        logger.info(
            "Entropia koefficians frissitve: %.4f -> %.4f",
            old_coef, new_coef,
        )

    def update_clip_epsilon(self, new_eps: float) -> None:
        old_eps: float = self.config.clip_epsilon
        self.config.clip_epsilon = new_eps
        logger.info("Clip epsilon frissitve: %.3f -> %.3f", old_eps, new_eps)

    def get_optimizer_state(self) -> dict[str, Any]:
        return self.optimizer.state_dict()

    def load_optimizer_state(self, state_dict: dict[str, Any]) -> None:
        self.optimizer.load_state_dict(state_dict)
        logger.info("Optimizer allapot visszaallitva.")
