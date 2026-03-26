"""
PPO Actor-Critic Neurális Hálózat Architektúra (networks.py).

Ez a modul a No-Limit Texas Hold'em RL AI "agyát" implementálja: egy
PyTorch alapú Actor-Critic hálózatot, amelyet a Proximal Policy
Optimization (PPO) algoritmus optimalizál.

Az architektúra felépítése:
    1. Beágyazó Rétegek (Embedding Layers):
       - CardEmbedding: 52-dim multi-hot -> card_embed_dim kompakt repr.
       - ContextEmbedding: Normalizált metrikák + pozíció -> context_embed_dim
       - HistoryEmbedding: 18x9 licittörténet -> history_embed_dim

    2. Fúziós Réteg: Beágyazott vektorok összefűzése (concatenation)

    3. Actor Fej (Policy Head):
       - MLP [512, 256, 128] -> 9-dimenziós logitok
       - Action Masking: logit += (1 - mask) * -1e8
       - Softmax -> Categorical eloszlás

    4. Critic Fej (Value Head):
       - MLP [512, 256, 128] -> 1-dimenziós skaláris V(s)

Architektúra szerződés:
    - Bemenet: Dict[str, Tensor] az ObservationBuilder-ből
    - Kimenet Actor: torch.distributions.Categorical (9 akció felett)
    - Kimenet Critic: Skaláris állapotérték V(s)

Hivatkozások:
    - Specifikáció: Neurális Architektúra Modul szekció
    - PPO: Schulman et al. (2017)
    - Ortogonális init: Saxe et al. (2014)
    - AlphaHoldem: AAAI 2022
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.distributions import Categorical

logger = logging.getLogger(__name__)


# =============================================================================
# Konfigurációs Adatosztály
# =============================================================================

@dataclass(frozen=True)
class NetworkConfig:
    """A teljes Actor-Critic hálózat konfigurációs paraméterei.

    Ezeket a config.yaml 'model' szekciójából tölti be a rendszer.

    Attributes:
        observation_dim: A laposított megfigyelési vektor teljes dimenziója.
        num_actions: Az akciótér mérete (9 a NLHE-ben).
        card_input_dim: Egyetlen kártyavektor dimenziója (52).
        context_input_dim: Környezeti metrikák dimenziója.
        history_input_dim: Licittörténet laposított dimenziója (18 * 9).
        position_input_dim: Pozíció one-hot vektor dimenziója.
        card_embed_dim: Kártyabeágyazás kimeneti dimenziója.
        context_embed_dim: Környezeti beágyazás kimeneti dimenziója.
        history_embed_dim: Történelem beágyazás kimeneti dimenziója.
        actor_hidden_layers: Az Actor MLP rejtett rétegeinek neuronszámai.
        critic_hidden_layers: A Critic MLP rejtett rétegeinek neuronszámai.
        activation: Aktivációs függvény neve ("relu", "gelu", "tanh").
        dropout: Dropout ráta a túlilleszkedés ellen.
        weight_init: Súlyinicializáció módja ("orthogonal", "xavier", "kaiming").
        weight_init_gain: Az ortogonális inicializáció gain paramétere.
        illegal_action_logit: Az illegális akciók logitjaihoz adott érték.
    """

    observation_dim: int = 281
    num_actions: int = 9
    card_input_dim: int = 52
    context_input_dim: int = 9
    history_input_dim: int = 162
    position_input_dim: int = 6
    card_embed_dim: int = 64
    context_embed_dim: int = 32
    history_embed_dim: int = 64
    actor_hidden_layers: tuple[int, ...] = (512, 256, 128)
    critic_hidden_layers: tuple[int, ...] = (512, 256, 128)
    activation: str = "relu"
    dropout: float = 0.1
    weight_init: str = "orthogonal"
    weight_init_gain: float = 1.0
    illegal_action_logit: float = -1.0e8

    def __post_init__(self) -> None:
        """Validálja a hálózat konfigurációs paramétereit."""
        if self.num_actions < 2:
            raise ValueError(f"num_actions legalább 2: kapott {self.num_actions}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout [0, 1) kell legyen: kapott {self.dropout}")
        if self.activation not in ("relu", "gelu", "tanh"):
            raise ValueError(f"Ismeretlen aktiváció: '{self.activation}'")
        logger.debug(
            "NetworkConfig: obs_dim=%d, actions=%d, actor=%s, init=%s",
            self.observation_dim, self.num_actions,
            self.actor_hidden_layers, self.weight_init,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any], num_players: int = None) -> "NetworkConfig":
        """
        Létrehoz egy NetworkConfig instance-t egy szótárból, szigorú mező
        szűréssel.

        Args:
            data: A konfigurációs szótár (általában a config.yaml-ból).
            num_players: A játékosok száma, ami felülírja a szótárban
                         található `num_players` értéket.

        Returns:
            NetworkConfig instance.
        """
        payload = data.copy()
        if num_players is not None:
            payload["num_players"] = num_players

        # Csak azokat a kulcsokat tartjuk meg, amik a dataclass mezői
        known_keys = cls.__dataclass_fields__.keys()
        filtered_payload = {k: v for k, v in payload.items() if k in known_keys}

        return cls(**filtered_payload)


# =============================================================================
# Segédfüggvények
# =============================================================================

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh,
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
        raise ValueError(f"Ismeretlen aktiváció: '{name}'. Elérhető: {list(_ACTIVATIONS.keys())}")
    return _ACTIVATIONS[name]()


def _build_mlp(
    input_dim: int,
    hidden_layers: tuple[int, ...] | list[int],
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.1,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    """Általános célú MLP építő: Linear -> Activation -> Dropout rétegekkel.

    Args:
        input_dim: Bemeneti vektor dimenziója.
        hidden_layers: Rejtett rétegek neuronszámai.
        output_dim: Kimeneti réteg dimenziója.
        activation: Aktivációs függvény neve.
        dropout: Dropout ráta (0.0 = nincs).
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
    """Kártyavektor beágyazó: 52-dim multi-hot -> kompakt tanult repr.

    Külön dolgozza fel a hole cards-ot és community cards-ot,
    majd összefűzi. Kimenet: card_embed_dim * 2.
    """

    def __init__(self, input_dim: int = 52, embed_dim: int = 64) -> None:
        super().__init__()
        self.hole_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim),
        )
        self.community_embed = nn.Sequential(
            nn.Linear(input_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim),
        )
        self.output_dim: int = embed_dim * 2
        logger.debug("CardEmbedding: %d -> %d (hole + community)", input_dim, self.output_dim)

    def forward(self, hole_cards: torch.Tensor, community_cards: torch.Tensor) -> torch.Tensor:
        """Forward: (batch, 52) x2 -> (batch, card_embed_dim*2)."""
        return torch.cat([self.hole_embed(hole_cards), self.community_embed(community_cards)], dim=-1)


class ContextEmbedding(nn.Module):
    """Környezeti metrikák + pozíció beágyazó."""

    def __init__(self, metrics_dim: int = 9, position_dim: int = 6, embed_dim: int = 32) -> None:
        super().__init__()
        total_input: int = metrics_dim + position_dim
        self.embed = nn.Sequential(
            nn.Linear(total_input, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, embed_dim),
        )
        self.output_dim: int = embed_dim
        logger.debug("ContextEmbedding: metrics(%d) + pos(%d) -> %d", metrics_dim, position_dim, embed_dim)

    def forward(self, env_metrics: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
        """Forward: (batch, metrics+pos) -> (batch, embed_dim)."""
        return self.embed(torch.cat([env_metrics, position], dim=-1))


class HistoryEmbedding(nn.Module):
    """Licittörténet beágyazó: flatten(18x9) -> kompakt repr."""

    def __init__(self, history_flat_dim: int = 162, embed_dim: int = 64) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(history_flat_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, embed_dim),
        )
        self.output_dim: int = embed_dim
        logger.debug("HistoryEmbedding: flat(%d) -> %d", history_flat_dim, embed_dim)

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
    (no parameter sharing), ami a PPO stabilitásának kulcsa.

    Example:
        >>> config = NetworkConfig(num_actions=9)
        >>> model = PokerActorCritic(config)
        >>> obs = {"hole_cards": torch.rand(32, 52), ...}
        >>> dist, value = model(obs)
        >>> action = dist.sample()
        >>> log_prob = dist.log_prob(action)
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        """Inicializálja az Actor-Critic hálózatot.

        Args:
            config: Hálózat konfiguráció. None = alapértelmezett NetworkConfig.
        """
        super().__init__()
        self.config: NetworkConfig = config or NetworkConfig()

        logger.info(
            "PokerActorCritic init: obs=%d, actions=%d, actor=%s, critic=%s, init=%s",
            self.config.observation_dim, self.config.num_actions,
            self.config.actor_hidden_layers, self.config.critic_hidden_layers,
            self.config.weight_init,
        )

        # --- Beágyazó Rétegek ---
        self.card_embedding = CardEmbedding(
            input_dim=self.config.card_input_dim, embed_dim=self.config.card_embed_dim,
        )
        self.context_embedding = ContextEmbedding(
            metrics_dim=self.config.context_input_dim,
            position_dim=self.config.position_input_dim,
            embed_dim=self.config.context_embed_dim,
        )
        self.history_embedding = HistoryEmbedding(
            history_flat_dim=self.config.history_input_dim,
            embed_dim=self.config.history_embed_dim,
        )

        # --- Fúziós dimenzió ---
        self._fusion_dim: int = (
            self.card_embedding.output_dim
            + self.context_embedding.output_dim
            + self.history_embedding.output_dim
        )
        logger.info(
            "Fúzió: cards=%d + ctx=%d + hist=%d = %d",
            self.card_embedding.output_dim, self.context_embedding.output_dim,
            self.history_embedding.output_dim, self._fusion_dim,
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
        train_p: int = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("PokerActorCritic kész: %s összes / %s tanítható param", f"{total_p:,}", f"{train_p:,}")

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def forward(
        self, observation: dict[str, torch.Tensor],
    ) -> tuple[Categorical, torch.Tensor]:
        """Teljes forward pass: obs dict -> (akció eloszlás, állapotérték).

        Lépések:
            1. Beágyazás: kártyák, kontextus, történelem
            2. Fúzió: összefűzés
            3. Actor: logitok + Action Masking + Softmax -> Categorical
            4. Critic: fúziós vektor -> skaláris V(s)

        Args:
            observation: Dict[str, Tensor] az ObservationBuilder-ből.
                Kulcsok: "hole_cards", "community_cards", "env_metrics",
                "position", "betting_history", "action_mask".

        Returns:
            Tuple (Categorical eloszlás, (batch,1) állapotérték).
        """
        hole_cards: torch.Tensor = observation["hole_cards"]
        community_cards: torch.Tensor = observation["community_cards"]
        env_metrics: torch.Tensor = observation["env_metrics"]
        position: torch.Tensor = observation["position"]
        betting_history: torch.Tensor = observation["betting_history"]
        action_mask: torch.Tensor = observation["action_mask"]

        # Auto batch dim
        is_single: bool = hole_cards.dim() == 1
        if is_single:
            hole_cards = hole_cards.unsqueeze(0)
            community_cards = community_cards.unsqueeze(0)
            env_metrics = env_metrics.unsqueeze(0)
            position = position.unsqueeze(0)
            betting_history = betting_history.unsqueeze(0)
            action_mask = action_mask.unsqueeze(0)

        batch_size: int = hole_cards.shape[0]
        logger.debug("Forward: batch=%d", batch_size)

        # 1. Beágyazás
        card_emb: torch.Tensor = self.card_embedding(hole_cards, community_cards)
        ctx_emb: torch.Tensor = self.context_embedding(env_metrics, position)
        hist_emb: torch.Tensor = self.history_embedding(betting_history)

        # 2. Fúzió
        fused: torch.Tensor = torch.cat([card_emb, ctx_emb, hist_emb], dim=-1)

        # 3. Actor: logitok + Action Masking
        logits: torch.Tensor = self.actor_head(fused)
        masked_logits: torch.Tensor = logits + (1.0 - action_mask) * self.config.illegal_action_logit

        # Biztonsági: üres maszk -> Fold kényszerítés
        valid_count: torch.Tensor = action_mask.sum(dim=-1)
        if (valid_count == 0).any():
            logger.error("KRITIKUS: %d minta üres maszkkal! Fold kényszerítve.",
                         int((valid_count == 0).sum().item()))
            empty_rows: torch.Tensor = valid_count == 0
            action_mask = action_mask.clone()
            action_mask[empty_rows, 0] = 1.0
            masked_logits = logits + (1.0 - action_mask) * self.config.illegal_action_logit

        # Softmax -> Categorical (numerikus stabilitás)
        action_probs: torch.Tensor = torch.softmax(masked_logits, dim=-1)
        action_probs = action_probs.clamp(min=1e-8)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        action_dist: Categorical = Categorical(probs=action_probs)

        # 4. Critic
        value: torch.Tensor = self.critic_head(fused)
        if is_single:
            value = value.squeeze(0)

        logger.debug("Forward kész: logits=[%.3f,%.3f], val=%.4f",
                      logits.min().item(), logits.max().item(), value.mean().item())

        return action_dist, value

    # =========================================================================
    # PPO Segédmetódusok
    # =========================================================================

    def evaluate_actions(
        self, observation: dict[str, torch.Tensor], actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Korábbi akciók kiértékelése az aktuális policy-vel (PPO update-hez).

        Args:
            observation: Batch-elt megfigyelés dict.
            actions: (batch,) korábbi akció indexek (long).

        Returns:
            Tuple: (log_probs, values, entropy) — mindegyik (batch,) vagy (batch,1).
        """
        action_dist, values = self.forward(observation)
        log_probs: torch.Tensor = action_dist.log_prob(actions)
        entropy: torch.Tensor = action_dist.entropy()

        logger.debug("evaluate_actions: batch=%d, lp=[%.4f,%.4f], ent=%.4f",
                      actions.shape[0], log_probs.min().item(), log_probs.max().item(),
                      entropy.mean().item())
        return log_probs, values, entropy

    def get_action(
        self, observation: dict[str, torch.Tensor], deterministic: bool = False,
    ) -> tuple[int, float, float]:
        """Egyetlen akció mintavételezés a rollout fázisban.

        Args:
            observation: Egyedi (nem batch) megfigyelés dict.
            deterministic: True = greedy, False = sztochasztikus.

        Returns:
            Tuple: (action_index, log_probability, state_value).
        """
        with torch.no_grad():
            action_dist, value = self.forward(observation)
            if deterministic:
                action: torch.Tensor = torch.argmax(action_dist.probs, dim=-1)
            else:
                action = action_dist.sample()
            log_prob: torch.Tensor = action_dist.log_prob(action)

        action_idx: int = int(action.reshape(-1)[0].item())
        lp_val: float = float(log_prob.reshape(-1)[0].item())
        v_val: float = float(value.reshape(-1)[0].item())

        logger.debug("get_action: idx=%d, lp=%.4f, v=%.4f, det=%s",
                      action_idx, lp_val, v_val, deterministic)
        return action_idx, lp_val, v_val

    def get_value(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Csak V(s) számítás (GAE bootstrap-hez).

        Args:
            observation: Egyedi vagy batch megfigyelés.

        Returns:
            Állapotérték tensor.
        """
        _, value = self.forward(observation)
        return value

    def save_checkpoint(self, path: str) -> None:
        """Menti a modell súlyait a megadott útvonalra.
        
        Args:
            path: A mentési útvonal.
        """
        torch.save(self.state_dict(), path)
        logger.info("Modell checkpoint mentve ide: %s", path)

    # =========================================================================
    # Súlyinicializáció
    # =========================================================================

    def _initialize_weights(self) -> None:
        """Lineáris rétegek inicializálása (orthogonal/xavier/kaiming).

        A policy output réteg kisebb gain-nel (0.01) inicializálódik
        a kezdeti exploráció elősegítésére. Bias vektorok: nulla.
        """
        init_method: str = self.config.weight_init
        gain: float = self.config.weight_init_gain
        count: int = 0

        for name, module in self.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            is_policy_out: bool = "actor_head" in name and module.out_features == self.config.num_actions
            g: float = 0.01 if is_policy_out else gain

            if init_method == "orthogonal":
                nn.init.orthogonal_(module.weight, gain=g)
            elif init_method == "xavier":
                nn.init.xavier_uniform_(module.weight, gain=g)
            elif init_method == "kaiming":
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            else:
                logger.warning("Ismeretlen init '%s', PyTorch default.", init_method)

            if module.bias is not None:
                nn.init.zeros_(module.bias)
            count += 1

        logger.info("Súlyinit: %s, gain=%.2f, %d Linear réteg (policy out gain=0.01)",
                     init_method, gain, count)

    # =========================================================================
    # Diagnosztika
    # =========================================================================

    def get_param_count(self) -> dict[str, int]:
        """Komponensenkénti paraméterszám.

        Returns:
            Dict: {komponens: count, ..., "total": N, "trainable": M}
        """
        c: dict[str, int] = {
            "card_embedding": sum(p.numel() for p in self.card_embedding.parameters()),
            "context_embedding": sum(p.numel() for p in self.context_embedding.parameters()),
            "history_embedding": sum(p.numel() for p in self.history_embedding.parameters()),
            "actor_head": sum(p.numel() for p in self.actor_head.parameters()),
            "critic_head": sum(p.numel() for p in self.critic_head.parameters()),
        }
        c["total"] = sum(c.values())
        c["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return c

    def summary(self) -> str:
        """Ember-olvasható architektúra összefoglaló."""
        c: dict[str, int] = self.get_param_count()
        return (
            "+" + "=" * 50 + "+\n"
            "|    PokerActorCritic Architektúra Összefoglaló    |\n"
            "+" + "=" * 50 + "+\n"
            f"|  Card Embedding:     {c['card_embedding']:>10,} params        |\n"
            f"|  Context Embedding:  {c['context_embedding']:>10,} params        |\n"
            f"|  History Embedding:  {c['history_embedding']:>10,} params        |\n"
            f"|  Actor (Policy):     {c['actor_head']:>10,} params        |\n"
            f"|  Critic (Value):     {c['critic_head']:>10,} params        |\n"
            "+" + "-" * 50 + "+\n"
            f"|  Total:              {c['total']:>10,} params        |\n"
            f"|  Trainable:          {c['trainable']:>10,} params        |\n"
            f"|  Fusion dim:         {self._fusion_dim:>10,}               |\n"
            f"|  Actions:            {self.config.num_actions:>10,}               |\n"
            f"|  Init:               {self.config.weight_init:>10}               |\n"
            "+" + "=" * 50 + "+"
        )
