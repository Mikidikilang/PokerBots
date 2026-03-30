"""
PPO Actor-Critic Neurális Hálózat Architektúra (networks.py).

Ez a modul a No-Limit Texas Hold'em RL AI "agyát" implementálja: egy
PyTorch alapú Actor-Critic hálózatot, amelyet a Proximal Policy
Optimization (PPO) algoritmus optimalizál.

Az architektúra felépítése:
    1. Beágyazó Rétegek (Embedding Layers):
       - CardEmbedding:    52-dim multi-hot   -> card_embed_dim  (x2: hole + community)
       - ContextEmbedding: env_metrics + pos   -> context_embed_dim
       - HistoryEmbedding: flatten(18x9)       -> history_embed_dim

    2. Fúziós Réteg: összefűzés  ->  trunk_input_dim

    3. Actor Fej (Policy Head):
       - MLP [512, 256, 128] -> 9-dimenziós logitok
       - Action Masking: torch.where + torch.finfo(dtype).min  (AMP-safe)
       - Softmax -> Categorical eloszlás

    4. Critic Fej (Value Head):
       - MLP [512, 256, 128] -> 1-dimenziós skaláris V(s)

Architektúra szerződés:
    - Bemenet:        Dict[str, Tensor]  az ObservationBuilder-ből
    - Kimenet Actor:  torch.distributions.Categorical  (9 akció felett)
    - Kimenet Critic: skaláris állapotérték V(s)

Javítások (v0.3.0):
    - Bug A: _initialize_weights() rekonstruálva (orphan kódfragmens eltávolítva)
    - Bug B: get_action_and_value() hozzáadva (collector + trainer közös API)
    - Bug C: NetworkConfig.from_dict() + trunk_input_dim / env_metrics_dim /
             position_dim / actor_hidden / critic_hidden property-k hozzáadva
    - Bug D: save_checkpoint() atomikus írással hozzáadva

Hivatkozások:
    - Specifikáció: Neurális Architektúra Modul szekció
    - PPO:          Schulman et al. (2017)
    - Ortogonális init: Saxe et al. (2014)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.env.sequential_history import LSTMHistoryEncoder

logger = logging.getLogger(__name__)


# =============================================================================
# Konfigurációs Adatosztály
# =============================================================================

@dataclass(frozen=True)
class NetworkConfig:
    """A teljes Actor-Critic hálózat konfigurációs paraméterei.

    Ezeket a config.yaml 'model' szekciójából tölti be a rendszer.
    A from_dict() factory metódus kezeli a YAML -> dataclass konverziót.

    Attributes:
        observation_dim:      A laposított megfigyelési vektor teljes dimenziója.
        num_actions:          Az akciótér mérete (9 a NLHE-ben).
        card_input_dim:       Egyetlen kártyavektor dimenziója (52).
        context_input_dim:    Környezeti metrikák dimenziója (4 + num_opponents).
        history_input_dim:    Licittörténet laposított dimenziója (18 * 9 = 162).
        position_input_dim:   Pozíció one-hot vektor dimenziója (== num_players).
        card_embed_dim:       Kártyabeágyazás kimeneti dimenziója.
        context_embed_dim:    Környezeti beágyazás kimeneti dimenziója.
        history_embed_dim:    Történelem beágyazás kimeneti dimenziója.
        actor_hidden_layers:  Az Actor MLP rejtett rétegeinek neuronszámai.
        critic_hidden_layers: A Critic MLP rejtett rétegeinek neuronszámai.
        activation:           Aktivációs függvény neve ("relu", "gelu", "tanh").
        dropout:              Dropout ráta a túlilleszkedés ellen.
        weight_init:          Súlyinicializáció módja.
        weight_init_gain:     Az ortogonális inicializáció gain paramétere.
        illegal_action_logit: Örökölt konstans – a kód torch.finfo(dtype).min-t
                              használ. Megtartva backward-compat céllal.
    """

    observation_dim: int = 335       # updated for 10 actions + 12-dim history
    num_actions: int = 10            # expanded from 9 (Priority-3 block bet fix)
    card_input_dim: int = 52
    context_input_dim: int = 9       # env_metrics_dim  (4 + num_opponents)
    history_input_dim: int = 216     # 18 * 12  (was 162 = 18*9, then 198 = 18*11)
    position_input_dim: int = 6      # == num_players
    card_embed_dim: int = 64
    context_embed_dim: int = 32
    history_embed_dim: int = 64
    actor_hidden_layers: tuple[int, ...] = (512, 256, 128)
    critic_hidden_layers: tuple[int, ...] = (512, 256, 128)
    activation: str = "relu"
    dropout: float = 0.1
    weight_init: str = "orthogonal"
    weight_init_gain: float = 1.0
    illegal_action_logit: float = -1.0e8  # DEPRECATED – lásd get_safe_mask_value

    # ------------------------------------------------------------------
    # Validáció
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validálja a hálózat konfigurációs paramétereit."""
        if self.num_actions < 2:
            raise ValueError(
                f"num_actions legalább 2 kell legyen, kapott: {self.num_actions}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                f"dropout [0, 1) tartományba kell esnie, kapott: {self.dropout}"
            )
        if self.activation not in ("relu", "gelu", "tanh"):
            raise ValueError(
                f"Ismeretlen aktiváció: '{self.activation}'. "
                f"Érvényes: relu | gelu | tanh"
            )
        logger.debug(
            "NetworkConfig: obs_dim=%d, actions=%d, actor=%s, init=%s",
            self.observation_dim,
            self.num_actions,
            self.actor_hidden_layers,
            self.weight_init,
        )

    # ------------------------------------------------------------------
    # Bug C FIX: Computed properties
    # ------------------------------------------------------------------

    @property
    def trunk_input_dim(self) -> int:
        """A fúziós vektor dimenziója: card*2 + context + history.

        Ez az Actor és Critic fejek bemeneti dimenziója.
        6-Max alapértelmezett: 64*2 + 32 + 64 = 224.
        """
        return (
            self.card_embed_dim * 2
            + self.context_embed_dim
            + self.history_embed_dim
        )

    @property
    def env_metrics_dim(self) -> int:
        """A környezeti metrikák dimenziója (pot, stack, call, raise + ellenfél stackek).

        Alias a context_input_dim-re, a test suite és a train_local.py kompatibilitásához.
        """
        return self.context_input_dim

    @property
    def position_dim(self) -> int:
        """A pozíció one-hot vektor dimenziója (== num_players).

        Alias a position_input_dim-re, a test suite kompatibilitásához.
        """
        return self.position_input_dim

    @property
    def actor_hidden(self) -> tuple[int, ...]:
        """Az Actor MLP rejtett rétegeinek neuronszámai.

        Alias az actor_hidden_layers-re, a test suite kompatibilitásához.
        """
        return self.actor_hidden_layers

    @property
    def critic_hidden(self) -> tuple[int, ...]:
        """A Critic MLP rejtett rétegeinek neuronszámai.

        Alias a critic_hidden_layers-re, a test suite kompatibilitásához.
        """
        return self.critic_hidden_layers

    # ------------------------------------------------------------------
    # Bug C FIX: Factory classmethod
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(
        cls,
        cfg: dict[str, Any],
        num_players: int = 6,
    ) -> NetworkConfig:
        """YAML konfigurációs szótárból példányosít.

        Kiszámítja az összes derived dimenziót (env_metrics, history, trunk)
        a konfigurációban definiált értékekből, így a YAML az egyetlen
        igazságforrás marad.

        Args:
            cfg:         A teljes config.yaml szótár.
            num_players: Az asztal játékosainak száma (pl. 6). A train_local.py
                         adja át a environment.num_players értékből.

        Returns:
            Teljesen inicializált NetworkConfig példány.

        Note:
            After the action-space expansion (9→10) and betting-history expansion
            (11→12 dims), existing checkpoints are incompatible. Training must
            restart from scratch.
        """
        env_cfg   = cfg.get("environment", {})
        obs_cfg   = env_cfg.get("observation_space", {})
        act_cfg   = env_cfg.get("action_space", {})
        model_cfg = cfg.get("model", {})
        embed_cfg = model_cfg.get("embedding", {})
        actor_cfg = model_cfg.get("actor", {})
        critic_cfg = model_cfg.get("critic", {})

        # --- Derived dimensions ---
        # Env metrics: pot_norm, my_chips_norm, amount_to_call_norm,
        #              min_raise_norm, pot_odds  +  one float per opponent stack
        # [PHASE 1] pot_odds added in ObservationBuilder._encode_env_metrics()
        env_metrics_dim: int = 5 + (num_players - 1)

        # Betting history: flat(max_actions × action_feature_dim)
        betting_history_dim: list[int] = obs_cfg.get("betting_history_dim", [18, 9])
        history_input_dim: int = betting_history_dim[0] * betting_history_dim[1]

        # Embedding dims
        card_embed_dim: int    = embed_cfg.get("card_embed_dim", 64)
        context_embed_dim: int = embed_cfg.get("context_embed_dim", 32)
        history_embed_dim: int = embed_cfg.get("history_embed_dim", 64)

        # Full observation dim (matches ObservationBuilder.get_observation_dim())
        # hole(52) + community(52) + env_metrics + history_flat + position
        card_dim: int        = obs_cfg.get("hole_cards_dim", 52) * 2
        position_dim: int    = num_players
        observation_dim: int = (
            card_dim
            + env_metrics_dim
            + history_input_dim
            + position_dim
        )

        # MLP layers
        actor_hidden: list[int]  = actor_cfg.get("hidden_layers", [512, 256, 128])
        critic_hidden: list[int] = critic_cfg.get("hidden_layers", [512, 256, 128])

        return cls(
            observation_dim=observation_dim,
            num_actions=act_cfg.get("num_actions", 9),
            card_input_dim=obs_cfg.get("hole_cards_dim", 52),
            context_input_dim=env_metrics_dim,
            history_input_dim=history_input_dim,
            position_input_dim=num_players,
            card_embed_dim=card_embed_dim,
            context_embed_dim=context_embed_dim,
            history_embed_dim=history_embed_dim,
            actor_hidden_layers=tuple(actor_hidden),
            critic_hidden_layers=tuple(critic_hidden),
            activation=actor_cfg.get("activation", "relu"),
            dropout=float(actor_cfg.get("dropout", 0.1)),
            weight_init=model_cfg.get("weight_init", "orthogonal"),
            weight_init_gain=float(model_cfg.get("weight_init_gain", 1.0)),
        )


# =============================================================================
# Segédfüggvények
# =============================================================================

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


def _get_activation(name: str) -> nn.Module:
    """Aktivációs függvényt hoz létre név alapján.

    Args:
        name: Az aktiváció neve ("relu", "gelu", "tanh").

    Returns:
        A megfelelő PyTorch aktivációs modul.

    Raises:
        ValueError: Ismeretlen név esetén.
    """
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Ismeretlen aktiváció: '{name}'. "
            f"Elérhető: {list(_ACTIVATIONS.keys())}"
        )
    return _ACTIVATIONS[name]()


def _build_mlp(
    input_dim: int,
    hidden_layers: tuple[int, ...] | list[int],
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.1,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Általános célú MLP builder: Linear -> Activation -> Dropout rétegekkel.

    Args:
        input_dim:        Bemeneti vektor dimenziója.
        hidden_layers:    Rejtett rétegek neuronszámai.
        output_dim:       Kimeneti réteg dimenziója.
        activation:       Aktivációs függvény neve.
        dropout:          Dropout ráta (0.0 = nincs dropout).
        final_activation: Opcionális kimeneti aktiváció.

    Returns:
        nn.Sequential modul.
    """
    layers: list[nn.Module] = []
    prev_dim: int = input_dim

    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(_get_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, output_dim))
    if final_activation is not None:
        layers.append(final_activation)

    return nn.Sequential(*layers)


# =============================================================================
# Beágyazó Modulok
# =============================================================================

class CardEmbedding(nn.Module):
    """Kártyavektor beágyazó: 52-dim multi-hot -> kompakt tanult reprezentáció.

    Külön dolgozza fel a hole cards-ot és community cards-ot,
    majd összefűzi. Kimenet: card_embed_dim * 2.
    """

    def __init__(
        self,
        input_dim: int = 52,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.hole_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.community_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.output_dim: int = embed_dim * 2
        logger.debug(
            "CardEmbedding: %d -> %d (hole + community)",
            input_dim,
            self.output_dim,
        )

    def forward(
        self,
        hole_cards: torch.Tensor,
        community_cards: torch.Tensor,
    ) -> torch.Tensor:
        """Forward: (batch, 52) x2 -> (batch, card_embed_dim*2)."""
        return torch.cat(
            [self.hole_embed(hole_cards), self.community_embed(community_cards)],
            dim=-1,
        )


class ContextEmbedding(nn.Module):
    """Környezeti metrikák + pozíció beágyazó."""

    def __init__(
        self,
        metrics_dim: int = 9,
        position_dim: int = 6,
        embed_dim: int = 32,
    ) -> None:
        super().__init__()
        total_input: int = metrics_dim + position_dim
        self.embed = nn.Sequential(
            nn.Linear(total_input, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.output_dim: int = embed_dim
        logger.debug(
            "ContextEmbedding: metrics(%d) + pos(%d) -> %d",
            metrics_dim,
            position_dim,
            embed_dim,
        )

    def forward(
        self,
        env_metrics: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """Forward: (batch, metrics+pos) -> (batch, embed_dim)."""
        return self.embed(torch.cat([env_metrics, position], dim=-1))


class HistoryEmbedding(nn.Module):
    """Licittörténet beágyazó: flatten(18x9) -> kompakt reprezentáció."""

    def __init__(
        self,
        history_flat_dim: int = 162,
        embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(history_flat_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.output_dim: int = embed_dim
        logger.debug(
            "HistoryEmbedding: flat(%d) -> %d",
            history_flat_dim,
            embed_dim,
        )

    def forward(self, betting_history: torch.Tensor) -> torch.Tensor:
        """Forward: (batch, 18, 9) vagy (batch, 162) -> (batch, embed_dim)."""
        if betting_history.dim() == 3:
            betting_history = betting_history.flatten(start_dim=1)
        return self.embed(betting_history)


# =============================================================================
# Fő Actor-Critic Hálózat
# =============================================================================

class PokerActorCritic(nn.Module):
    """PPO Actor-Critic hálózat No-Limit Texas Hold'em pókerhez.

    Három fő komponens:
        1. Beágyazó rétegek: kártyák, kontextus, történelem -> fúziós vektor
        2. Actor (Policy) fej: fúziós vektor -> maszkolt Categorical eloszlás
        3. Critic (Value) fej: fúziós vektor -> skaláris V(s)

    Az Actor és Critic fejek a beágyazás után SZÉTVÁLNAK
    (no parameter sharing a fejekben), ami a PPO stabilitásának kulcsa
    tökéletlen információs játékokban.

    Egyetlen belépési pont a collector és a trainer számára:
        get_action_and_value(obs, action=None) -> (action, log_prob, entropy, value)

    Convergence note (Priority-4 documentation fix):
        The EMA/FSP average-strategy network reduces exploitability and
        cycle-avoidance in multi-player self-play. However, theoretical Nash
        Equilibrium convergence guarantees (Heinrich & Silver 2015) apply
        only to two-player zero-sum games. 6-Max NLHE is neither; the
        avg_network approximates an exploitability-reducing average strategy,
        not a formal Nash Equilibrium.

    Example:
        >>> config = NetworkConfig.from_dict(yaml_cfg, num_players=6)
        >>> model  = PokerActorCritic(config).to(device)
        >>> # Rollout fázis (nincs gradient)
        >>> with torch.inference_mode():
        ...     action, lp, ent, val = model.get_action_and_value(obs)
        >>> # PPO update fázis (gradient kell)
        >>> lp, val, ent = model.evaluate_actions(obs, stored_actions)
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        """Inicializálja az Actor-Critic hálózatot.

        Args:
            config: Hálózat konfiguráció. None = alapértelmezett NetworkConfig.
        """
        super().__init__()
        self.config: NetworkConfig = config or NetworkConfig()

        logger.info(
            "PokerActorCritic init: obs=%d, actions=%d, "
            "actor=%s, critic=%s, init=%s",
            self.config.observation_dim,
            self.config.num_actions,
            self.config.actor_hidden_layers,
            self.config.critic_hidden_layers,
            self.config.weight_init,
        )

        # --- Beágyazó Rétegek ---
        self.card_embedding = CardEmbedding(
            input_dim=self.config.card_input_dim,
            embed_dim=self.config.card_embed_dim,
        )
        self.context_embedding = ContextEmbedding(
            metrics_dim=self.config.context_input_dim,
            position_dim=self.config.position_input_dim,
            embed_dim=self.config.context_embed_dim,
        )
        self.history_encoder = LSTMHistoryEncoder(
            action_feature_dim=13,  # Phase 2.5: betting history is [18, 13]
            hidden_dim=256,
            num_layers=2,
            dropout=0.1,
            bidirectional=False,
        )

        # --- Fúziós dimenzió (== cards*2 + context + history_encoder) ---
        self._fusion_dim: int = (
            self.card_embedding.output_dim
            + self.context_embedding.output_dim
            + self.history_encoder.output_dim
        )
        logger.info(
            "Fúzió: cards=%d + ctx=%d + lstm_hist=%d = %d",
            self.card_embedding.output_dim,
            self.context_embedding.output_dim,
            self.history_encoder.output_dim,
            self._fusion_dim,
        )

        # --- Actor (Policy) Fej ---
        self.actor_head: nn.Sequential = _build_mlp(
            input_dim=self._fusion_dim,
            hidden_layers=self.config.actor_hidden_layers,
            output_dim=self.config.num_actions,
            activation=self.config.activation,
            dropout=self.config.dropout,
        )

        # --- Critic (Value) Fej ---
        self.critic_head: nn.Sequential = _build_mlp(
            input_dim=self._fusion_dim,
            hidden_layers=self.config.critic_hidden_layers,
            output_dim=1,
            activation=self.config.activation,
            dropout=self.config.dropout,
        )

        # --- Súlyinicializáció ---
        self._initialize_weights()

        total_p: int = sum(p.numel() for p in self.parameters())
        train_p: int = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        logger.info(
            "PokerActorCritic kész: %s összes / %s tanítható param",
            f"{total_p:,}",
            f"{train_p:,}",
        )

    # =========================================================================
    # Bug A FIX: _initialize_weights() — teljes, helyes implementáció
    # =========================================================================

    def _initialize_weights(self) -> None:
        """Ortogonális (vagy Xavier/Kaiming) súlyinicializáció.

        Konvenciók:
            - Minden Linear réteg: orthogonal_(gain=weight_init_gain)
            - Actor kimeneti réteg: orthogonal_(gain=0.01)
              -> közel egyenletes kezdeti policy eloszlás
            - Critic kimeneti réteg: orthogonal_(gain=1.0)
              -> normális értékbecslési tartomány
            - Minden bias: zeros_()

        A gain=0.01 az Actor output rétegen kritikus: túl nagy kezdeti
        logit-különbségek esetén a policy már az első iterációban
        összeomlana egy akcióra, megakadályozva a felfedezést.
        """
        init_method: str = self.config.weight_init
        gain: float = self.config.weight_init_gain
        count: int = 0

        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if init_method == "orthogonal":
                nn.init.orthogonal_(module.weight, gain=gain)
            elif init_method == "xavier":
                nn.init.xavier_uniform_(module.weight, gain=gain)
            elif init_method == "kaiming":
                nn.init.kaiming_uniform_(
                    module.weight, nonlinearity="relu"
                )
            else:
                # Fallback: ortogonális
                nn.init.orthogonal_(module.weight, gain=gain)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            count += 1

        # --- Actor kimeneti réteg: kis gain a közel-egyenletes kezdeti policyhoz ---
        # Az actor_head utolsó eleme mindig a Linear(128, 9) kimeneti réteg.
        actor_output_layer = self._get_output_layer(self.actor_head)
        if actor_output_layer is not None:
            nn.init.orthogonal_(actor_output_layer.weight, gain=0.01)
            if actor_output_layer.bias is not None:
                nn.init.zeros_(actor_output_layer.bias)
            logger.debug(
                "Actor output réteg újrainicializálva: orthogonal(gain=0.01)"
            )

        # --- Critic kimeneti réteg: gain=1.0 (normális értéktartomány) ---
        critic_output_layer = self._get_output_layer(self.critic_head)
        if critic_output_layer is not None:
            nn.init.orthogonal_(critic_output_layer.weight, gain=1.0)
            if critic_output_layer.bias is not None:
                nn.init.zeros_(critic_output_layer.bias)
            logger.debug(
                "Critic output réteg újrainicializálva: orthogonal(gain=1.0)"
            )

        logger.info(
            "Súlyinicializáció: %s, gain=%.2f, %d Linear réteg "
            "(actor_out gain=0.01, critic_out gain=1.0)",
            init_method,
            gain,
            count,
        )

    @staticmethod
    def _get_output_layer(sequential: nn.Sequential) -> nn.Linear | None:
        """Visszaadja az nn.Sequential utolsó Linear rétegét.

        Args:
            sequential: Egy _build_mlp() által épített nn.Sequential.

        Returns:
            Az utolsó nn.Linear modul, vagy None ha nem található.
        """
        last_linear: nn.Linear | None = None
        for module in sequential.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        return last_linear

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(
        self,
        observation: dict[str, torch.Tensor],
    ) -> tuple[Categorical, torch.Tensor]:
        """Teljes forward pass: obs dict -> (akció eloszlás, állapotérték).

        Lépések:
            1. Beágyazás: kártyák, kontextus, történelem
            2. Fúzió:     összefűzés -> trunk vektor
            3. Actor:     logitok + Action Masking (AMP-safe) + Softmax -> Categorical
            4. Critic:    trunk -> skaláris V(s)

        Phase 4-21: Unified forward API always expects and returns batched tensors.
        The collector always provides batched inputs (with batch dimension).

        Args:
            observation: Dict[str, Tensor] az ObservationBuilder-ből.
                Kötelező kulcsok: all (batch, ...):
                    "hole_cards"       (batch, 52)
                    "community_cards"  (batch, 52)
                    "env_metrics"      (batch, N)
                    "position"         (batch, P)
                    "betting_history"  (batch, 18, 13)  [PHASE 2: LSTM input, NOT flattened]
                    "action_mask"      (batch, 9)

        Returns:
            Tuple (Categorical eloszlás, (batch, 1) állapotérték tenzor).
        """
        hole_cards:      torch.Tensor = observation["hole_cards"]
        community_cards: torch.Tensor = observation["community_cards"]
        env_metrics:     torch.Tensor = observation["env_metrics"]
        position:        torch.Tensor = observation["position"]
        betting_history: torch.Tensor = observation["betting_history"]
        action_mask:     torch.Tensor = observation["action_mask"]

        batch_size: int = hole_cards.shape[0]
        logger.debug("Forward: batch=%d", batch_size)

        # 1. Beágyazás
        card_emb: torch.Tensor = self.card_embedding(hole_cards, community_cards)
        ctx_emb:  torch.Tensor = self.context_embedding(env_metrics, position)
        # [PHASE 2] LSTM History Encoder: DO NOT flatten, pass (batch, 18, 13) directly
        hist_emb: torch.Tensor = self.history_encoder(betting_history)

        # 2. Fúzió
        fused: torch.Tensor = torch.cat([card_emb, ctx_emb, hist_emb], dim=-1)

        # 3. Actor: logitok + AMP-safe Action Masking
        logits: torch.Tensor = self.actor_head(fused)

        # dtype-aware maszk érték: float16-ban -1e8 NaN-t okozna
        mask_value: float = torch.finfo(logits.dtype).min
        masked_logits: torch.Tensor = torch.where(
            action_mask.bool(),
            logits,
            torch.tensor(mask_value, dtype=logits.dtype, device=logits.device),
        )

        # Biztonsági ellenőrzés: üres maszk -> ValueError
        valid_count: torch.Tensor = action_mask.sum(dim=-1)
        if (valid_count == 0).any():
            empty_rows = valid_count == 0
            n_empty = int(empty_rows.sum().item())
            logger.critical(
                "KRITIKUS: %d minta üres maszkkal! Ez egy parseléshiba vagy környezet inkonzisztencia.",
                n_empty,
            )
            raise ValueError(
                f"Empty action mask detected for {n_empty} samples. "
                "This indicates a parsing failure or environment inconsistency in RTA/inference context."
            )

        # Numerikusan stabil Softmax -> valószínűség eloszlás
        # Pass masked_logits directly to Categorical; let PyTorch handle
        # the near-negative-infinity values numerically safely with log_softmax
        action_dist: Categorical = Categorical(logits=masked_logits)

        # 4. Critic
        value: torch.Tensor = self.critic_head(fused)

        logger.debug(
            "Forward kész: logit_range=[%.3f, %.3f], val_mean=%.4f",
            logits.min().item(),
            logits.max().item(),
            value.mean().item(),
        )

        return action_dist, value

    # =========================================================================
    # Bug B FIX: get_action_and_value() — egységes API a collector és trainer számára
    # =========================================================================

    def get_action_and_value(
        self,
        observation: dict[str, torch.Tensor],
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Egységes belépési pont a rollout gyűjtéshez és a PPO frissítéshez.

        Két működési mód:
            Inference (action=None):
                A policy-ből mintavételez egy akciót, visszaadja annak
                log-valószínűségét, az eloszlás entrópiaját és V(s)-t.
                Ezt a collector.py hívja torch.inference_mode()-ban.

            Evaluation (action!=None):
                A megadott (korábban tárolt) akciót értékeli ki az aktuális
                policy-vel. Visszaadja az akció log-valószínűségét, az
                eloszlás entrópiaját és V(s)-t.
                Ezt a trainer.py hívja a gradient frissítési lépésben.

        Visszatérési értékek sorrendje a collector és trainer hívásokkal
        kompatibilis:
            action, log_prob, entropy, value  ->  4-tuple

        Args:
            observation: Dict[str, Tensor] az ObservationBuilder-ből.
            action:      Ha None, mintavételez. Ha Tensor (batch,), kiértékeli.
                         A trainer long() típusú action indexeket ad át.

        Returns:
            Tuple of:
                action   (batch,)   – mintavételezett vagy átadott akcióindex
                log_prob (batch,)   – az akcio log-valószínűsége a policy alatt
                entropy  (batch,)   – policy eloszlás entrópiája
                value    (batch, 1) – állapotérték V(s)
        """
        action_dist, value = self.forward(observation)

        if action is None:
            # Rollout fázis: sztochasztikus mintavételezés
            action = action_dist.sample()
        else:
            # PPO update fázis: a tárolt akciót értékeljük ki
            # Biztosítjuk a helyes alakot és típust
            action = action.long().reshape(-1)

        log_prob: torch.Tensor = action_dist.log_prob(action)
        entropy:  torch.Tensor = action_dist.entropy()

        logger.debug(
            "get_action_and_value: action=%s, lp_mean=%.4f, ent_mean=%.4f",
            action.shape,
            log_prob.mean().item(),
            entropy.mean().item(),
        )

        return action, log_prob, entropy, value

    # =========================================================================
    # evaluate_actions() — kényelmes wrapper a PPO update-hez
    # =========================================================================

    def evaluate_actions(
        self,
        observation: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Korábbi akciók kiértékelése az aktuális policy-vel (PPO update-hez).

        Ez a metódus a get_action_and_value() thin wrapper-e, a régi
        call-site kompatibilitásának megőrzéséhez.

        Args:
            observation: Batch-elt megfigyelés dict.
            actions:     (batch,) korábbi akció indexek (long).

        Returns:
            Tuple: (log_probs, values, entropy) — mindegyik (batch,) vagy (batch,1).
        """
        _, log_probs, entropy, values = self.get_action_and_value(
            observation, action=actions
        )
        logger.debug(
            "evaluate_actions: batch=%d, lp_range=[%.4f, %.4f], ent=%.4f",
            actions.shape[0],
            log_probs.min().item(),
            log_probs.max().item(),
            entropy.mean().item(),
        )
        return log_probs, values, entropy

    # =========================================================================
    # get_action() — egyedi akció mintavételezés (rollout fázis, single step)
    # =========================================================================

    def get_action(
        self,
        observation: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Egyetlen akció mintavételezés a rollout fázisban.

        [RTA-3 FIX] Auto-unsqueeze: if the input tensors are unbatched (no batch
        dimension), a batch dimension of 1 is automatically added before forward().
        This makes get_action() safe to call from both the collector (which already
        batches) and from RTA inference code (which typically passes single states).

        Args:
            observation:   Observation dict. May be batched (batch, ...) or
                           unbatched (...). Batch dimension is auto-detected and
                           added if missing.
            deterministic: True = greedy argmax, False = stochastic sample.

        Returns:
            Tuple: (action_index, log_probability, state_value).
        """
        # [RTA-3 FIX] Auto-detect and correct missing batch dimension.
        # hole_cards / community_cards: unbatched = (52,), batched = (B, 52)
        # betting_history: unbatched = (18, 12), batched = (B, 18, 12)
        # position / env_metrics / action_mask: unbatched = (D,), batched = (B, D)
        _needs_unsqueeze = False
        for _k, _v in observation.items():
            if _k == "betting_history":
                _needs_unsqueeze = (_v.dim() == 2)
            else:
                _needs_unsqueeze = (_v.dim() == 1)
            break  # Check first key only — all tensors have the same batch structure

        if _needs_unsqueeze:
            observation = {k: v.unsqueeze(0) for k, v in observation.items()}

        # Phase 4-23: Use strict inference_mode for better safety and performance
        with torch.inference_mode():
            action_dist, value = self.forward(observation)
            if deterministic:
                action: torch.Tensor = torch.argmax(
                    action_dist.probs, dim=-1
                )
            else:
                action = action_dist.sample()
            log_prob: torch.Tensor = action_dist.log_prob(action)

        action_idx: int   = int(action.reshape(-1)[0].item())
        lp_val: float     = float(log_prob.reshape(-1)[0].item())
        v_val: float      = float(value.reshape(-1)[0].item())

        logger.debug(
            "get_action: idx=%d, lp=%.4f, v=%.4f, det=%s",
            action_idx, lp_val, v_val, deterministic,
        )
        return action_idx, lp_val, v_val

    # =========================================================================
    # get_value() — csak V(s) számítás (GAE bootstrap-hez)
    # =========================================================================

    def get_value(
        self, observation: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Csak V(s) számítás (GAE bootstrap-hez a rollout végén).

        Args:
            observation: Egyedi vagy batch megfigyelés.

        Returns:
            Állapotérték tenzor.
        """
        _, value = self.forward(observation)
        return value

    # =========================================================================
    # Bug D FIX: save_checkpoint() — atomikus írással
    # =========================================================================

    def save_checkpoint(
        self,
        filepath: str,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        """A modell súlyait checkpoint fájlba menti atomikus írással.

        Az atomikus írás garantálja, hogy a CommitScheduler soha nem olvas
        részlegesen megírt fájlt: a torch.save() egy ideiglenes .tmp fájlba
        ír, majd os.replace() végzi az atomi átnevezést. POSIX garantálja,
        hogy az átnevezés atomikus, ezért nincs lehetőség sérült checkpoint-ra.

        Args:
            filepath:    A cél .pt fájl elérési útja.
            extra_state: Opcionális extra szótár (pl. optimizer state,
                         iteration, total_steps). Össze lesz olvasztva
                         a model_state_dict-tel.

        Raises:
            RuntimeError: Ha a checkpoint írása valamilyen okból meghiúsul.
                          A .tmp fájl automatikusan törlődik.

        Example:
            >>> model.save_checkpoint(
            ...     "checkpoints/checkpoint_iter_01000.pt",
            ...     extra_state={
            ...         "optimizer_state_dict": opt.state_dict(),
            ...         "iteration": 1000,
            ...     },
            ... )
        """
        checkpoint: dict[str, Any] = {
            "model_state_dict": self.state_dict(),
        }
        if extra_state is not None:
            # extra_state kulcsai felülírják a checkpoint-ot ha ütköznek
            checkpoint.update(extra_state)

        # Atomi írás: temp fájl -> végleges
        target_dir: str = os.path.dirname(os.path.abspath(filepath))
        os.makedirs(target_dir, exist_ok=True)
        tmp_path: str = filepath + ".tmp"

        try:
            torch.save(checkpoint, tmp_path)
            # POSIX garantálja: os.replace() atomi
            os.replace(tmp_path, filepath)
        except Exception as exc:
            # Temp fájl takarítása hiba esetén
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise RuntimeError(
                f"Atomikus checkpoint mentés meghiúsult: {filepath}"
            ) from exc

        # Fájlméret logolás
        try:
            size_mb: float = os.path.getsize(filepath) / (1024 * 1024)
        except OSError:
            size_mb = 0.0

        logger.info(
            "Checkpoint mentve: %s (%.2f MB)",
            filepath,
            size_mb,
        )

    # =========================================================================
    # Diagnosztika
    # =========================================================================

    def get_param_count(self) -> dict[str, int]:
        """Komponensenkénti paraméterszám.

        Returns:
            Dict: {komponens: count, ..., "total": N, "trainable": M}
        """
        counts: dict[str, int] = {
            "card_embedding":    sum(p.numel() for p in self.card_embedding.parameters()),
            "context_embedding": sum(p.numel() for p in self.context_embedding.parameters()),
            "history_embedding": sum(p.numel() for p in self.history_embedding.parameters()),
            "actor_head":        sum(p.numel() for p in self.actor_head.parameters()),
            "critic_head":       sum(p.numel() for p in self.critic_head.parameters()),
        }
        counts["total"]     = sum(counts.values())
        counts["trainable"] = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        return counts

    def summary(self) -> str:
        """Ember-olvasható architektúra összefoglaló."""
        c = self.get_param_count()
        return (
            "+" + "=" * 52 + "+\n"
            "|     PokerActorCritic Architektúra Összefoglaló     |\n"
            "+" + "=" * 52 + "+\n"
            f"|  Card Embedding:      {c['card_embedding']:>10,} params          |\n"
            f"|  Context Embedding:   {c['context_embedding']:>10,} params          |\n"
            f"|  History Embedding:   {c['history_embedding']:>10,} params          |\n"
            f"|  Actor  (Policy):     {c['actor_head']:>10,} params          |\n"
            f"|  Critic (Value):      {c['critic_head']:>10,} params          |\n"
            "+" + "-" * 52 + "+\n"
            f"|  Total:               {c['total']:>10,} params          |\n"
            f"|  Trainable:           {c['trainable']:>10,} params          |\n"
            f"|  Trunk dim:           {self._fusion_dim:>10,}                 |\n"
            f"|  Actions:             {self.config.num_actions:>10,}                 |\n"
            f"|  Weight init:         {self.config.weight_init:>10}                 |\n"
            "+" + "=" * 52 + "+"
        )
