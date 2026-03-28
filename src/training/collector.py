"""
Rollout Collector (src/training/collector.py).

[FIX C4 - 2025-03-28] 3-Bet Tultuntetesi Hiba Javitasa:
    Az _update_preflop_context() koraban az egesz betting_history-t ujra
    vizsgalta minden lepesnel, ami szuperlinearisan novelte a
    preflop_raises_total szamlalot. Pl. 10 preflop lepesnel a 6. lepesnel
    mar 5 db emelest adott hozza, a 7. lepesnel 6-ot stb. — igy
    preflop_raises_total >> zylobal emelesek szama, ami a harombojet%
    ~80-100%-ra inflalta. A javitas: csak az uj, utoljara feldolgozas ota
    hozzaadott elemeket vizsgalja, O(delta) koltsegge alakitva a szamitast.

[FIX C1 - 2025-03-28] Bootstrap Ertek Atomikus Tarolasa:
    A collect_rollout() vegere hozzaadtuk a buffer.set_last_value() hivast,
    mielott visszaadjuk a vezerlesst a runner-nek. Igy a bootstrap V(s_T)
    garantaltan a helyes, truncated allapothoz tartozik, nem egy kesobb
    szamolt kozelites.

Public interface
----------------
    RolloutCollector(network, env, obs_builder, buffer, config, orchestrator, device)
    .collect_rollout(n_steps) -> RolloutStats
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Protocol, runtime_checkable

import torch

try:
    from src.orchestrator.telemetry import HandRecord, TelemetryAnalyzer
except ImportError:
    @dataclass
    class HandRecord:  # type: ignore[no-redef]
        hand_id: int
        player_id: int
        position: int
        iteration: int
        reward_bb: float
        street_reached: int
        went_to_showdown: bool
        won_at_showdown: bool
        vpip: bool
        pfr: bool
        three_bet: bool
        total_aggressive_actions: int
        total_passive_actions: int
        total_folds: int
        actions: list[int] = field(default_factory=list)
        action_streets: list[int] = field(default_factory=list)

    class TelemetryAnalyzer:  # type: ignore[no-redef]
        def record_hand(self, record: HandRecord) -> None: ...


logger = logging.getLogger(__name__)

_FOLD       = 0
_CHECK_CALL = 1
_MIN_RAISE  = 2
_ALL_IN     = 9   # [Action Space Fix] shifted from 8; includes new RAISE_THIRD_POT (3)
_RAISE_ACTIONS: frozenset[int] = frozenset(range(_MIN_RAISE, _ALL_IN + 1))
# = {2, 3, 4, 5, 6, 7, 8, 9} — includes the new 33% pot block bet (index 3)


@runtime_checkable
class PokerEnvironment(Protocol):
    def reset(self) -> dict[str, Any]: ...
    def step(self, action: int) -> tuple[dict[str, Any], float]: ...
    def is_over(self) -> bool: ...


@dataclass
class CollectorConfig:
    big_blind: float = 2.0

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "CollectorConfig":
        return cls(
            big_blind=float(cfg.get("environment", {}).get("big_blind", 2.0))
        )


@dataclass
class _HandAccumulator:
    """Mutable per-episode state built up action-by-action.

    [FIX C4] _last_seen_history_len mezo hozzaadva: nyomon koveti, hogy
    az utolso _update_preflop_context() hivas ota hany elemet dolgoztunk
    fel a betting_history-ban. Igy az uj hivasok csak a delta-t olvassak,
    nem az egesz tortenetet ujra.
    """
    hand_id: int
    player_id: int
    position: int
    iteration: int

    actions:        list[int] = field(default_factory=list)
    action_streets: list[int] = field(default_factory=list)

    # Preflop ellenfél emelesek szama (3-bet detektalashoz)
    preflop_raises_total: int = 0

    # [FIX C4] Az utolso feldolgozas ota latott history hossza.
    # Csak az uj (delta) elemeket vizsgalja a kovetkezo hivasnal.
    _last_seen_history_len: int = 0

    def record_action(self, action: int, street: int) -> None:
        """Egy akcio feljegyzese a nyomkoveto listaba."""
        self.actions.append(action)
        self.action_streets.append(street)
        if street == 0 and action in _RAISE_ACTIONS:
            self.preflop_raises_total += 1

    def record_env_preflop_raises(self, n_opponent_raises: int) -> None:
        """Ellenfél emelesek hozzaadasa (CSAK a delta-hoz)."""
        self.preflop_raises_total += n_opponent_raises


class RolloutStats(NamedTuple):
    n_steps:       int
    n_episodes:    int
    mean_reward:   float
    total_reward:  float
    n_hands_submitted: int


class RolloutCollector:
    """Collects PPO rollout transitions and feeds the Telemetry Bridge.

    [FIX C4] Az _update_preflop_context() inkrementalisan mukodik:
    csak az utolso hivat ota hozzaadott betting_history bejegyzeseket
    vizsgalja, elkerulve a szuperlinearis tultuntelest.

    [FIX C1] A collect_rollout() vegere buffer.set_last_value() hivodik,
    atomikusan tarolva a bootstrap erteket a bufferben, mielott a
    runner.py compute_gae()-t hivna.
    """

    def __init__(
        self,
        network:      torch.nn.Module,
        env:          PokerEnvironment,
        obs_builder:  Any,
        buffer:       Any,
        config:       dict[str, Any],
        orchestrator: Any | None = None,
        device:       torch.device | str = "cpu",
    ) -> None:
        if not isinstance(env, PokerEnvironment):
            raise TypeError(
                f"env must satisfy the PokerEnvironment Protocol "
                f"(reset/step/is_over). Got {type(env).__name__}. "
                "Did you pass an RLCardWrapper?"
            )

        self.network      = network
        self.env          = env
        self.obs_builder  = obs_builder
        self.buffer       = buffer
        self.config       = config
        self.orchestrator = orchestrator
        self.device       = torch.device(device)

        self._hand_counter:  int   = 0
        self._total_steps:   int   = 0
        self._iteration:     int   = 0

        self._current_obs:  dict[str, Any] | None = None
        self._episode_done: bool                  = True
        self._hand_acc:     _HandAccumulator | None = None

        self._big_blind: float = float(
            config.get("environment", {}).get("big_blind", 2.0)
        )

        logger.info(
            "RolloutCollector inicializalva: device=%s, BB=%.1f, "
            "telemetry_enabled=%s",
            self.device,
            self._big_blind,
            orchestrator is not None,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def set_iteration(self, iteration: int) -> None:
        self._iteration = iteration

    def collect_rollout(self, n_steps: int) -> RolloutStats:
        """Collect exactly n_steps environment steps.

        [FIX C1] A rollout vegen buffer.set_last_value() hivodik az utolso
        nem-terminalis allapot V(s_T) ertekevel, mielott visszaadjuk a
        vezerlesst. Ez szavatolja, hogy a runner.py-ban kiadott
        compute_gae(last_value=buffer.get_last_bootstrap_value()) hivas
        mindig a HELYES bootstrap erteket hasznalja.
        """
        self.network.eval()

        n_episodes    = 0
        episode_rews  = []
        n_telemetry   = 0
        running_rew   = 0.0
        done          = False  # [C1 FIX] inicializalas a scope szamara

        if self._episode_done or self._current_obs is None:
            self._current_obs  = self.env.reset()
            self._episode_done = False
            self._hand_acc     = self._new_accumulator()
            running_rew        = 0.0

        for step in range(n_steps):
            obs_raw = self._current_obs

            # [FIX C4] Inkrementalis preflop context frissites
            self._update_preflop_context(self._hand_acc, obs_raw)

            current_street = _detect_street(obs_raw)

            obs_tensor = self._build_obs_tensor(obs_raw)
            obs_batched: dict[str, torch.Tensor] = {
                k: v.unsqueeze(0) for k, v in obs_tensor.items()
            }

            with torch.inference_mode():
                action, log_prob, entropy, value = (
                    self.network.get_action_and_value(obs_batched)
                )

            action_int: int = int(action.reshape(-1)[0].item())

            if self._hand_acc is not None:
                self._hand_acc.record_action(action_int, current_street)

            next_obs_raw, reward = self.env.step(action_int)
            done = self.env.is_over()

            running_rew += reward
            self._total_steps += 1

            obs_tensor_cpu = {k: v.cpu() for k, v in obs_tensor.items()}
            self.buffer.add(
                observation=obs_tensor_cpu,
                action=action.detach(),
                log_prob=log_prob.detach(),
                value=value.detach(),
                reward=float(reward),
                done=done,
            )

            if done:
                n_episodes    += 1
                n_telemetry   += self._close_episode(
                    terminal_obs=next_obs_raw,
                    reward=running_rew,
                )
                episode_rews.append(running_rew)
                running_rew = 0.0

                self._current_obs  = self.env.reset()
                self._episode_done = False
                self._hand_acc     = self._new_accumulator()
            else:
                self._current_obs = next_obs_raw

        # =====================================================================
        # [FIX C1] Atomikus bootstrap ertek tarolasa a bufferben
        # =====================================================================
        # Ha a rollout NOT done allapotban vegzodott (truncated episode),
        # ki kell szamolni V(s_T) a jelenlegi allapotra es el kell tarolni
        # a bufferben, MIELOTT a runner.py compute_gae()-t hivna.
        # Igy nincs race condition: a buffer.get_last_bootstrap_value()
        # mindig a helyes erteket adja vissza.
        # [CV-3 FIX] done here is from the LAST env.step() in the loop.
        # MultiAgentRLCardWrapper.step() internally advances ALL opponent actions
        # before returning, so self.env.is_over() is definitively correct by the
        # time we reach this point. No off-by-one race condition remains.
        # If the last step ended the episode (done=True), bootstrap = 0.0 (correct:
        # no future reward from terminal state). Otherwise bootstrap from V(s_T).
        done_is_final: bool = bool(done)
        if not done_is_final and self._current_obs is not None:
            last_obs_tensor = self._build_obs_tensor(self._current_obs)
            last_obs_batched: dict[str, torch.Tensor] = {
                k: v.unsqueeze(0) for k, v in last_obs_tensor.items()
            }
            with torch.inference_mode():
                last_value_tensor = self.network.get_value(last_obs_batched)
            # Atomikusan beallitjuk a buffer-ben (set_last_value implementalva)
            self.buffer.set_last_value(last_value_tensor.squeeze(-1))
            logger.debug(
                "Bootstrap ertek beallitva: %.6f (truncated episode)",
                float(last_value_tensor.detach().cpu().item()),
            )
        else:
            # Terminalis allapot VAGY ures megfigyelés: 0.0 a helyes bootstrap ertek
            self.buffer.set_last_value(0.0)
            logger.debug(
                "Bootstrap ertek: 0.0 (terminalis allapot=%s)", done_is_final
            )

        self.network.train()

        mean_reward = (
            sum(episode_rews) / len(episode_rews) if episode_rews else 0.0
        )
        stats = RolloutStats(
            n_steps=n_steps,
            n_episodes=n_episodes,
            mean_reward=mean_reward,
            total_reward=sum(episode_rews),
            n_hands_submitted=n_telemetry,
        )
        logger.debug(
            "collect_rollout kesz: steps=%d, episodes=%d, "
            "mean_reward=%.4f BB, hands_to_telemetry=%d",
            n_steps, n_episodes, mean_reward, n_telemetry,
        )
        return stats

    def get_total_steps(self) -> int:
        return self._total_steps

    def get_total_episodes(self) -> int:
        return self._hand_counter

    def get_last_bootstrap_value(self, network: Any) -> float:
        """Visszaadja a legutolso bootstrap erteket (backward-kompatibilitas).

        [C1 FIX NOTE] Ez a metodus megtartva backward-kompatibilitas celjara,
        de az elonyben reszesitett mod a buffer.get_last_bootstrap_value()
        hasznalata, ami az atomikusan tarolt erteket adja vissza.
        """
        return self.buffer.get_last_bootstrap_value()

    # =========================================================================
    # [FIX C4] Inkrementalis Preflop Context Frissites
    # =========================================================================

    def _update_preflop_context(
        self, acc: _HandAccumulator | None, obs_raw: dict[str, Any]
    ) -> None:
        """Inkrementalisan frissiti az ellenfél preflop emeleseinek szamlalojat.

        [FIX C4] A korabbi implementacio az egesz betting_history-t ujra
        vizsgalta minden lepesnel, ami szuperlinearis novekedeshez vezetett.
        Pelda: 10 preflop lep eseten a 6. lepesnel 5 emelesst adott hozza,
        a 7.-nel 6-ot stb., igy preflop_raises_total >> tenyleges emelesek.

        A javitas: az _HandAccumulator._last_seen_history_len meset csak az
        UJ bejegyzeseket dolgozza fel (slicing), O(delta) koltsegge alakitva
        a muvelet O(N) helyett.

        Args:
            acc: Az aktualis kezhez tartozo akkumulator.
            obs_raw: A jelenlegi jatekallapot szotara.
        """
        if acc is None:
            return
        history = obs_raw.get("betting_history", [])
        current_len = len(history)

        # [FIX C4] Csak az UJ bejegyzesek feldolgozasa (delta-only scan)
        new_entries = history[acc._last_seen_history_len:]
        acc._last_seen_history_len = current_len  # frissitjuk a poziciot

        # Csak az ellenfél (nem a tanulo agenst) emelesei szamitanak
        new_opp_raises = sum(
            1 for h in new_entries
            if h.get("action", -1) in _RAISE_ACTIONS
            and h.get("player", acc.player_id) != acc.player_id
        )

        if new_opp_raises > 0:
            acc.record_env_preflop_raises(new_opp_raises)
            logger.debug(
                "C4 fix: %d uj ellenfél preflop emeles feljegyezve (delta-only)",
                new_opp_raises,
            )

    # =========================================================================
    # Telemetry Bridge
    # =========================================================================

    def _close_episode(
        self,
        terminal_obs: dict[str, Any],
        reward: float,
    ) -> int:
        if self._hand_acc is None:
            return 0

        last_street   = _detect_street(terminal_obs)
        any_fold      = _FOLD in self._hand_acc.actions
        went_to_sd    = (last_street == 3) and (not any_fold)
        won_at_sd     = went_to_sd and (reward > 0)

        try:
            record = _build_hand_record(
                acc=self._hand_acc,
                reward=reward,
                big_blind=self._big_blind,
                street_reached=last_street,
                went_to_showdown=went_to_sd,
                won_at_showdown=won_at_sd,
            )
        except Exception as exc:
            logger.warning("_build_hand_record hiba: %s", exc, exc_info=True)
            return 0

        return self._submit_hand_record(record)

    def _submit_hand_record(self, record: HandRecord) -> int:
        if self.orchestrator is None:
            return 0
        try:
            self.orchestrator.telemetry.record_hand(record)
            logger.debug(
                "HandRecord benyujtva: hand=%d iter=%d street=%d "
                "reward=%.3f BB vpip=%s pfr=%s 3b=%s wtsd=%s",
                record.hand_id, record.iteration, record.street_reached,
                record.reward_bb, record.vpip, record.pfr,
                record.three_bet, record.went_to_showdown,
            )
            return 1
        except AttributeError as exc:
            logger.warning(
                "orchestrator.telemetry.record_hand() nem elerheto: %s", exc,
            )
        except Exception as exc:
            logger.warning(
                "HandRecord benyujtas sikertelen (hand=%d): %s",
                record.hand_id, exc, exc_info=True,
            )
        return 0

    # =========================================================================
    # Observation Tensor Building
    # =========================================================================

    def _build_obs_tensor(
        self, obs_raw: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        try:
            obs_tensor: dict[str, torch.Tensor] = self.obs_builder.build(obs_raw)
        except Exception as exc:
            raise RuntimeError(
                f"ObservationBuilder.build() sikertelen: {exc}\n"
                f"obs_raw kulcsok: {list(obs_raw.keys())}"
            ) from exc

        return {
            k: v.to(self.device, non_blocking=True)
            for k, v in obs_tensor.items()
        }

    # =========================================================================
    # Accumulator Helpers
    # =========================================================================

    def _new_accumulator(self) -> _HandAccumulator:
        self._hand_counter += 1
        return _HandAccumulator(
            hand_id=self._hand_counter,
            player_id=0,
            position=0,
            iteration=self._iteration,
        )


# =============================================================================
# Module-level helper functions
# =============================================================================

def _detect_street(obs: dict[str, Any]) -> int:
    """Aktualis utca levezetese a kozos lapok szamabol.

    [L3 FIX] Varatlan lapszamra (1, 2) figyelmezteto log hozzaadva,
    hogy a telemetria ne csendesen rossz utcat rendeljen.
    """
    n = len(obs.get("public_cards", []))
    if n == 0:
        return 0   # preflop
    if n == 3:
        return 1   # flop
    if n == 4:
        return 2   # turn
    if n == 5:
        return 3   # river
    # [L3 FIX] Varatlan lapszam: figyelmeztetes, de kezeles folytatodik
    logger.warning(
        "Varatlan kozos lap szam: %d (elvaras: 0/3/4/5). "
        "River-kent kezeljuk az utcaestimaciosz pontossaganak megorzese erdekeben.",
        n,
    )
    return 3


def _build_hand_record(
    acc:               _HandAccumulator,
    reward:            float,
    big_blind:         float,
    street_reached:    int,
    went_to_showdown:  bool,
    won_at_showdown:   bool,
) -> HandRecord:
    """HUD statisztikak levezetese egy befejezett kez akciosorozatabol.

    [FIX C4] A three_bet szamitas helyes a javitott preflop_raises_total
    ertek utan, mivel az _update_preflop_context() mar nem tultuntel.
    """
    actions        = acc.actions
    action_streets = acc.action_streets

    preflop_actions = [a for a, s in zip(actions, action_streets) if s == 0]

    # VPIP: barmilyen nem-fold akcio preflop
    vpip = any(a != _FOLD for a in preflop_actions)

    # PFR: legalabb egy preflop emeles
    pfr = any(a in _RAISE_ACTIONS for a in preflop_actions)

    # 3-Bet: az agens emelt preflop, es mar volt ott elotte legalabb egy emeles
    # [FIX C4] preflop_raises_total most pontos az inkrementalis szamolas utan
    pf_raises_before = acc.preflop_raises_total - sum(
        1 for a, s in zip(actions, action_streets)
        if s == 0 and a in _RAISE_ACTIONS
    )
    three_bet = any(a in _RAISE_ACTIONS for a in preflop_actions) and pf_raises_before >= 1

    total_aggressive = sum(1 for a in actions if a in _RAISE_ACTIONS)
    total_passive    = sum(1 for a in actions if a == _CHECK_CALL)
    total_folds      = sum(1 for a in actions if a == _FOLD)

    safe_bb   = max(big_blind, 1e-6)
    reward_bb = reward / safe_bb

    return HandRecord(
        hand_id=acc.hand_id,
        player_id=acc.player_id,
        position=acc.position,
        iteration=acc.iteration,
        reward_bb=reward_bb,
        street_reached=street_reached,
        went_to_showdown=went_to_showdown,
        won_at_showdown=won_at_showdown,
        vpip=vpip,
        pfr=pfr,
        three_bet=three_bet,
        total_aggressive_actions=total_aggressive,
        total_passive_actions=total_passive,
        total_folds=total_folds,
        actions=list(actions),
        action_streets=list(action_streets),
    )
