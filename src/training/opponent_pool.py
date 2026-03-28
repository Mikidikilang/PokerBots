"""
Ellenfel-Pool Kezelo (opponent_pool.py).

A Fictitious Self-Play (FSP) es a curriculum rendszer ellenfel-keszletenek
kezeleset vegzi. Ket fo feladatot lat el:

    1. Statikus Archetipusok: Egyszerusitett, heurisztika-alapu botok
       (Calling Station, Maniac, Random, Tight Passive) amelyek a
       Phase 0 betanitashoz szuksegesek.

    2. FSP Snapshot Pool: A tanulo halozat korabbi allapotainak
       (checkpointjainak) gyujtemenye, amelyekbol a MAB algoritmus
       valaszt ellenfelet a Phase 2-ben.

Hivatkozasok:
    - Specifikacio: opponent_pool.py — korabbi sulyok es statikus archetipusok
    - Curriculum: Phase 0 (static bots), Phase 2 (FSP + MAB)
"""

from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from src.model.networks import NetworkConfig, PokerActorCritic

logger = logging.getLogger(__name__)


# =============================================================================
# Absztrakt Ellenfel Interfesz
# =============================================================================

class OpponentAgent(ABC):
    """Absztrakt bazisosztalya minden ellenfel agensnek.

    Minden ellenfelnek (statikus bot vagy korabbi snapshot)
    implementalnia kell a select_action metodust.
    """

    @abstractmethod
    def select_action(
        self, legal_actions: list[int], game_state: dict[str, Any]
    ) -> int:
        """Kivalaszt egy akciot a legalis akciok kozul.

        Args:
            legal_actions: Az ervenyes akcio indexek listaja (0-8).
            game_state: Az aktualis jatekszituacio szotarja.

        Returns:
            A kivalasztott akcio indexe.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Az ellenfel azonosito neve."""
        ...


# =============================================================================
# Statikus Archetipusok (Phase 0)
# =============================================================================

class CallingStationBot(OpponentAgent):
    """Passziv bot: szinte mindig megadja a tetet (Check/Call).

    A Calling Station nem emel, nem bloffol, es nagyon ritkan dob.
    Fo feladata az All-in Spam patologia ellenszerenek biztositasa:
    mivel minden All-in-t megad, a tulzott bloffolest buntetni tudja.
    """

    @property
    def name(self) -> str:
        return "calling_station"

    def select_action(
        self, legal_actions: list[int], game_state: dict[str, Any]
    ) -> int:
        """Mindig Check/Call-t valaszt ha lehetseges, egyebkent Fold.

        Args:
            legal_actions: Ervenyes akciok.
            game_state: Jatekallapot.

        Returns:
            Akcio index (1=Check/Call vagy 0=Fold).
        """
        if 1 in legal_actions:  # Check/Call
            return 1
        if 0 in legal_actions:  # Fold (biztonsagi fallback)
            return 0
        return legal_actions[0] if legal_actions else 0


class ManiacBot(OpponentAgent):
    """Hiper-agressziv bot: folyamatosan emel es All-in-ozik.

    A Maniac maximalis nyomast helyez a passziv agensre. Fo feladata
    a passzivitas patologia ellenszerenek biztositasa: a vakok
    ellopasaval kikenyszeriti az agressziv vedekezest.
    """

    @property
    def name(self) -> str:
        return "maniac"

    def select_action(
        self, legal_actions: list[int], game_state: dict[str, Any]
    ) -> int:
        """A leheto legagresszivebb akciot valasztja.

        Prioritas: All-in(8) > Raise 2x(7) > Raise 1.5x(6) > ... > Check/Call(1)

        Args:
            legal_actions: Ervenyes akciok.
            game_state: Jatekallapot.

        Returns:
            A legmagasabb indexu (legagresszivebb) legalis akcio.
        """
        # Csokkeno prioritas az agresszio alapjan
        priority: list[int] = [8, 7, 6, 5, 4, 3, 2, 1, 0]
        for action in priority:
            if action in legal_actions:
                return action
        return legal_actions[0] if legal_actions else 0


class RandomBot(OpponentAgent):
    """Teljesen veletlenszeru bot: egyenletes valoszinuseggel valaszt.

    A Random bot a felfedezesi fazisban hasznalja az Orchestrator,
    ahol a cel a jatekszabalyok es az alapveto strategia elsajatitasa.
    """

    @property
    def name(self) -> str:
        return "random"

    def select_action(
        self, legal_actions: list[int], game_state: dict[str, Any]
    ) -> int:
        """Egyenletes valoszinuseggel valaszt a legalis akciok kozul.

        Args:
            legal_actions: Ervenyes akciok.
            game_state: Jatekallapot.

        Returns:
            Veletlenszeruen kivalasztott akcio index.
        """
        return random.choice(legal_actions) if legal_actions else 0


class TightPassiveBot(OpponentAgent):
    """Szoros-passziv bot: csak premium kezekkel jatszik, es nem emel.

    A Tight Passive bot a "nit" archetipust kepviseli: keveset jatszik
    (alacsony VPIP), de amit jatszik, azt passzivan (alacsony PFR).
    Fo feladata a jatekos bloffoles-tanulasanak elosegitese.
    """

    def __init__(self, play_frequency: float = 0.15) -> None:
        """Inicializalas.

        Args:
            play_frequency: A jatekba szallas valoszinusege (0.0-1.0).
        """
        self._play_freq: float = play_frequency

    @property
    def name(self) -> str:
        return "tight_passive"

    def select_action(
        self, legal_actions: list[int], game_state: dict[str, Any]
    ) -> int:
        """Az esetek tobbsegeben Fold-ot valaszt, egyebkent Check/Call-t.

        Args:
            legal_actions: Ervenyes akciok.
            game_state: Jatekallapot.

        Returns:
            Akcio index.
        """
        if random.random() > self._play_freq:
            if 0 in legal_actions:
                return 0  # Fold
        if 1 in legal_actions:
            return 1  # Check/Call
        return legal_actions[0] if legal_actions else 0


class NeuralOpponentAgent(OpponentAgent):
    """Neuralis halozat alapu ellenfel: SFT (Supervised Fine-Tuned) model.

    Ez az osztaly egy korabbi betanitott PokerActorCritic halozatot
    kapszulaz az OpponentAgent interfeszbe, es lehetove teszi neuralis
    ellenfeleink hasznalatat a curriculum tanulasban (Phase 1-2).

    A halozat determinisztikus (greedy) akciovasztagassal fog jatszani,
    hogy stabilabb es konzisztensebb kuzdemi stilust biztosítson.

    Attributes:
        _name: Az ellenfel azonositoneve.
        _network: A PokerActorCritic halozat peldanya.
        _obs_builder: Az ObservationBuilder a nyersszabalyok objektumok konvertalasahoz.
        _device: A PyTorch device (cpu/cuda).
    """

    def __init__(
        self,
        name: str,
        network: PokerActorCritic,
        obs_builder: Any,
        device: torch.device | str = "cpu",
    ) -> None:
        """Inicializalja a neuralis ellenfelet.

        Args:
            name: Az ellenfel azonosito neve (pl. 'sft_aggressive').
            network: PokerActorCritic halozat betoltve es kialakitva.
            obs_builder: ObservationBuilder peldany a jatekallpot konvertalasahoz.
            device: PyTorch device (cpu/cuda).
        """
        self._name: str = name
        self._network: PokerActorCritic = network
        self._obs_builder: Any = obs_builder
        self._device: torch.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )

    @property
    def name(self) -> str:
        """Az ellenfel azonositoneve."""
        return self._name

    def select_action(
        self, legal_actions: list[int], game_state: dict[str, Any]
    ) -> int:
        """Kivalaszt egy akciot a neuralis halozat alapjan.

        A metodus:
            1. A nyersszabalyok jatekallpotot megfigyelesekke konvertalja.
            2. Batch dimenziot ad hozza (unsqueeze).
            3. Meghivja a network.get_action() halozatot determinisztikus módban.
            4. Ha a valasztott akcio nem legalis, biztonsagi fallback-re lephet.

        Args:
            legal_actions: Az ervenyes akcio indexek listaja (0-8).
            game_state: Az aktualis jatekszituacio szotarja.

        Returns:
            Az ActionIdx, ami az legal_actions-ben van, vagy biztonsagi fallback.
        """
        try:
            # 1. Jatekallpot konvertalasa megfigyelesekke
            obs_dict: dict[str, torch.Tensor] = self._obs_builder.build(game_state)

            # 2. Batch dimenzio hozzaadasa (unsqueeze)
            # A halozat (batch_size, ...) alakok vár
            batched_obs: dict[str, torch.Tensor] = {}
            for key, tensor in obs_dict.items():
                # Feltételezzük, hogy obs_dict nem-batch alakú:
                # tensor shape: (d,) -> unsqueeze -> (1, d)
                batched_obs[key] = tensor.unsqueeze(0).to(self._device)

            # 3. Determinisztikus akcio mintavételezés
            action_idx, _, _ = self._network.get_action(
                batched_obs, deterministic=True
            )

            # 4. Biztonsagi ellenorzes: az akcio legalis-e?
            if action_idx in legal_actions:
                return action_idx

            # Fallback: ha az akcio nem legalis, valasszunk valami biztosabbat
            logger.warning(
                "NeuralOpponentAgent (%s): action_idx=%d not in legal_actions=%s. "
                "Falling back to safe action.",
                self._name, action_idx, legal_actions,
            )

            # Prioritas: Check/Call(1) > Fold(0) > egyeb
            if 1 in legal_actions:
                return 1
            if 0 in legal_actions:
                return 0
            return legal_actions[0] if legal_actions else 0

        except Exception as e:
            logger.error(
                "NeuralOpponentAgent (%s): neueralis akciovasztagas hiba: %s. "
                "Fallback: Check/Call vagy random.",
                self._name, e,
            )
            # Vasaltsag fallback: Check/Call vagy random
            if 1 in legal_actions:
                return 1
            return random.choice(legal_actions) if legal_actions else 0


# =============================================================================
# Archetipus Registry
# =============================================================================

_ARCHETYPE_REGISTRY: dict[str, type[OpponentAgent]] = {
    "calling_station": CallingStationBot,
    "maniac": ManiacBot,
    "random": RandomBot,
    "tight_passive": TightPassiveBot,
}


def create_archetype(name: str, **kwargs: Any) -> OpponentAgent:
    """Letrehoz egy statikus archetipus botot vagy neuralis SFT ellenfelet nev alapjan.

    Ha a nev egy SFT bot nev (sft_aggressive, sft_balanced, sft_passive) es a
    szukseges parameterek (network_config, obs_builder, device) megadva vannak,
    megprobalja betolteni az elofeldolgozva halozat sulyait a
    checkpoints/sft/{name}.pt fajlbol.

    Ha a szukseges parameterek hianyoznak vagy a weights fajl nem letezik,
    logol egy WARNING-et es fallback-re lepkedik az egyenerteku statikus bothoz:
        - sft_aggressive -> ManiacBot
        - sft_balanced   -> RandomBot
        - sft_passive    -> TightPassiveBot

    Args:
        name: Az archetipus vagy SFT bot neve.
        **kwargs: Extra parameterek, beleertve:
            - network_config: NetworkConfig peldany (SFT botokhoz szukseges).
            - obs_builder: ObservationBuilder peldany (SFT botokhoz szukseges).
            - device: torch.device vagy str (SFT botokhoz szukseges).

    Returns:
        OpponentAgent peldany.

    Raises:
        ValueError: Ha a nev ismeretlen es nem SFT bot.
    """
    # SFT bot szotara: nev -> (fallback_static_bot_name, sft_weights_path)
    _SFT_BOTS: dict[str, str] = {
        "sft_aggressive": "maniac",
        "sft_balanced": "random",
        "sft_passive": "tight_passive",
    }

    # Ellerzetse a nev SFT bot
    if name in _SFT_BOTS:
        network_config: Any = kwargs.get("network_config")
        obs_builder: Any = kwargs.get("obs_builder")
        device: Any = kwargs.get("device", "cpu")

        if network_config is not None and obs_builder is not None:
            try:
                # SFT weights betoltes
                from src.model.networks import PokerActorCritic

                # Halozat letrehozasa
                network: PokerActorCritic = PokerActorCritic(network_config)
                if isinstance(device, str):
                    device = torch.device(device)
                network = network.to(device)

                # Súlyok betoltes checkpoints/sft/{name}.pt-bol
                sft_weights_path: Path = Path("checkpoints/sft") / f"{name}.pt"
                if sft_weights_path.exists():
                    state_dict: dict[str, Any] = torch.load(
                        sft_weights_path, map_location=device, weights_only=True
                    )
                    network.load_state_dict(state_dict)
                    logger.info("SFT weights betoltve: %s (%s)", name, sft_weights_path)

                    # Neuralis ellenfel letrehozasa
                    agent: OpponentAgent = NeuralOpponentAgent(
                        name=name,
                        network=network,
                        obs_builder=obs_builder,
                        device=device,
                    )
                    logger.info("NeuralOpponentAgent letrehozva: %s", name)
                    return agent
                else:
                    # Súlyok fajl nem letezik - fallback
                    fallback_name: str = _SFT_BOTS[name]
                    logger.warning(
                        "SFT weights fajl nem letezik: %s. "
                        "Fallback az statikus bothoz: %s",
                        sft_weights_path, fallback_name,
                    )
                    return create_archetype(fallback_name, **kwargs)

            except Exception as e:
                # Hiba a neuralis ellenfel letrehozasaban - fallback
                fallback_name = _SFT_BOTS[name]
                logger.warning(
                    "SFT ellenfel letrehozasar hiba: %s. "
                    "Fallback az statikus bothoz: %s — %s",
                    name, fallback_name, e,
                )
                return create_archetype(fallback_name, **kwargs)
        else:
            # Szukseges parameterek hianyoznak - fallback
            fallback_name = _SFT_BOTS[name]
            logger.warning(
                "SFT bot '%s' inicializalasahoz szukseges parameterek hianyoznak "
                "(network_config, obs_builder, device). "
                "Fallback az statikus bothoz: %s",
                name, fallback_name,
            )
            return create_archetype(fallback_name, **kwargs)

    # Statikus archetipusok kezelese
    if name not in _ARCHETYPE_REGISTRY:
        raise ValueError(
            f"Ismeretlen archetipus: '{name}'. "
            f"Elerheto: {list(_ARCHETYPE_REGISTRY.keys()) + list(_SFT_BOTS.keys())}"
        )
    agent = _ARCHETYPE_REGISTRY[name](**kwargs)
    logger.debug("Archetipus letrehozva: %s", name)
    return agent


# =============================================================================
# Snapshot Kezelo (FSP)
# =============================================================================

@dataclass
class PoolSnapshot:
    """Egy korabbi halozati allapot metaadatai a pool-ban.

    Attributes:
        snapshot_id: Egyedi azonosito.
        filepath: A checkpoint fajl eleresi utja.
        iteration: A training iteracio szama a mentes idejen.
        win_rate: Az elert nyeresi rata a mentes idejen (mbb/hand).
        selection_count: Hanyszor lett kivalasztva ellenfelkent.
        total_reward: Az osszesitett jutalom a vele jatszott partikbol.
    """

    snapshot_id: str = ""
    filepath: str = ""
    iteration: int = 0
    win_rate: float = 0.0
    selection_count: int = 0
    total_reward: float = 0.0


class OpponentPool:
    """Az ellenfel-keszlet teljes eletciklus-kezeloje.

    Egyesiti a statikus archetipusokat es a dinamikus FSP
    snapshot-okat egyetlen egységes pool-ba, amelybol a
    CurriculumManager (MAB algoritmussal) valaszt ellenfelet.

    Attributes:
        archetypes: Statikus botok szotarja.
        snapshots: FSP snapshot-ok listaja.
        max_pool_size: Maximalis snapshot-ok szama.
    """

    def __init__(
        self,
        archetype_names: list[str] | None = None,
        max_pool_size: int = 20,
        snapshot_dir: str = "checkpoints/snapshots",
        network_config: NetworkConfig | None = None,
        obs_builder: Any | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Inicializalja az ellenfel-pool-t.

        Args:
            archetype_names: A betoltendo statikus archetipusok nevei (pl. statikusak vagy SFT botok).
            max_pool_size: Maximalis snapshot-ok szama (FIFO rotacio).
            snapshot_dir: A snapshot fajlok konyvtara.
            network_config: NetworkConfig peldany (SFT botok inicializalasahoz).
            obs_builder: ObservationBuilder peldany (SFT botok inicializalasahoz).
            device: PyTorch device (cpu/cuda) SFT botok inicializalasahoz.
        """
        self.max_pool_size: int = max_pool_size
        self.snapshot_dir: Path = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

        # Szemelyzetesít device-t
        if isinstance(device, str):
            device = torch.device(device)
        self.device: torch.device = device

        # Statikus archetipusok es SFT botok betoltese
        names: list[str] = archetype_names or [
            "calling_station", "maniac", "random", "tight_passive"
        ]
        self.archetypes: dict[str, OpponentAgent] = {}

        # Parameterek az SFT botokhoz
        sft_kwargs: dict[str, Any] = {
            "network_config": network_config,
            "obs_builder": obs_builder,
            "device": device,
        }

        for name in names:
            try:
                self.archetypes[name] = create_archetype(name, **sft_kwargs)
            except ValueError as e:
                logger.error("Archetipus betoltes sikertelen: %s", e)

        # FSP snapshot pool
        self.snapshots: list[PoolSnapshot] = []

        logger.info(
            "OpponentPool inicializalva: %d archetipus, max_snapshots=%d, device=%s",
            len(self.archetypes), self.max_pool_size, self.device,
        )

    # =========================================================================
    # Archetipus Kezeles
    # =========================================================================

    def get_archetype(self, name: str) -> OpponentAgent | None:
        """Visszaad egy statikus archetipust nev alapjan.

        Args:
            name: Az archetipus neve.

        Returns:
            OpponentAgent peldany vagy None ha nem talalhato.
        """
        return self.archetypes.get(name)

    def get_all_archetype_names(self) -> list[str]:
        """Visszaadja az osszes elerheto archetipus nevet.

        Returns:
            Archetipus nevek listaja.
        """
        return list(self.archetypes.keys())

    # =========================================================================
    # FSP Snapshot Kezeles
    # =========================================================================

    def add_snapshot(
        self,
        model_state_dict: dict[str, Any],
        iteration: int,
        win_rate: float = 0.0,
    ) -> str:
        """Uj halozati allapot-pillanatfelvételt ad a pool-hoz.

        Ha a pool eleri a maximalis meretet, a legregibbet torolja (FIFO).

        Args:
            model_state_dict: A halozat state_dict() kimenete.
            iteration: Az aktualis training iteracio szama.
            win_rate: Az aktualis nyeresi rata.

        Returns:
            Az uj snapshot egyedi azonositoja.
        """
        snapshot_id: str = f"snapshot_iter_{iteration:06d}"
        filepath: str = str(self.snapshot_dir / f"{snapshot_id}.pt")

        # Mentes diszkre
        torch.save(model_state_dict, filepath)

        snapshot = PoolSnapshot(
            snapshot_id=snapshot_id,
            filepath=filepath,
            iteration=iteration,
            win_rate=win_rate,
        )

        # FIFO rotacio
        if len(self.snapshots) >= self.max_pool_size:
            removed: PoolSnapshot = self.snapshots.pop(0)
            logger.debug("Snapshot eltavolitva (FIFO): %s", removed.snapshot_id)

        self.snapshots.append(snapshot)

        logger.info(
            "Snapshot hozzaadva: %s (iter=%d, wr=%.2f, pool=%d/%d)",
            snapshot_id, iteration, win_rate,
            len(self.snapshots), self.max_pool_size,
        )
        return snapshot_id

    def load_snapshot_weights(self, snapshot_id: str) -> dict[str, Any] | None:
        """Betolti egy snapshot sulyait a diszkrol.

        Args:
            snapshot_id: A snapshot egyedi azonositoja.

        Returns:
            A state_dict szotar, vagy None ha nem talalhato.
        """
        for snap in self.snapshots:
            if snap.snapshot_id == snapshot_id:
                try:
                    # Phase 3-18: Use weights_only=True for secure deserialization.
                    # Snapshots contain only network state_dicts (weights), safe to load.
                    state_dict: dict[str, Any] = torch.load(
                        snap.filepath, map_location="cpu", weights_only=True
                    )
                    snap.selection_count += 1
                    logger.debug("Snapshot betoltve: %s", snapshot_id)
                    return state_dict
                except Exception as e:
                    logger.error("Snapshot betoltes hiba: %s — %s", snapshot_id, e)
                    return None
        logger.warning("Snapshot nem talalhato: %s", snapshot_id)
        return None

    def get_snapshot_ids(self) -> list[str]:
        """Visszaadja az osszes snapshot azonositot.

        Returns:
            Snapshot ID-k listaja.
        """
        return [s.snapshot_id for s in self.snapshots]

    # =========================================================================
    # Egyesitett Pool Muveletek
    # =========================================================================

    def get_all_opponent_names(self) -> list[str]:
        """Visszaadja az osszes elerheto ellenfel nevet (archetipusok + snapshots).

        Returns:
            Ellenfel nevek listaja.
        """
        names: list[str] = list(self.archetypes.keys())
        names.extend(s.snapshot_id for s in self.snapshots)
        return names

    def get_pool_size(self) -> int:
        """Az ellenfel-pool teljes merete (archetipusok + snapshots).

        Returns:
            A pool elemeinek szama.
        """
        return len(self.archetypes) + len(self.snapshots)

    def select_random_opponent(self) -> str:
        """Veletlenszeruen valaszt egy ellenfelet a teljes pool-bol.

        Returns:
            A kivalasztott ellenfel neve/azonositoja.
        """
        all_names: list[str] = self.get_all_opponent_names()
        if not all_names:
            logger.warning("Ures ellenfel-pool! Fallback: random bot.")
            return "random"
        selected: str = random.choice(all_names)
        logger.debug("Veletlenszeru ellenfel kivalasztva: %s", selected)
        return selected

    def get_pool_stats(self) -> dict[str, Any]:
        """Visszaadja a pool osszefoglalo statisztikait.

        Returns:
            Dict a pool allapotaval.
        """
        return {
            "num_archetypes": len(self.archetypes),
            "num_snapshots": len(self.snapshots),
            "total_pool_size": self.get_pool_size(),
            "snapshot_iterations": [s.iteration for s in self.snapshots],
            "snapshot_selection_counts": [s.selection_count for s in self.snapshots],
        }
