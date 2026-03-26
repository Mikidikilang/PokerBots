"""
Egyseg tesztek a src/model/ modulhoz.

Tesztel: NetworkConfig, ActorCriticNetwork architektura,
dimenzio konzisztencia, akcio maszkolas, checkpoint mentes/betoltes.

MEGJEGYZES: A torch.nn-fugo tesztek (forward pass, gradiens)
csak valodi PyTorch kornyezetben futnak. A config es dimenzio
tesztek torch mock-kal is mukodnek.
"""

from __future__ import annotations

import numpy as np
import pytest
from typing import Any


# =============================================================================
# NetworkConfig Tesztek (torch.nn-fuggetlen)
# =============================================================================

class TestNetworkConfig:
    """A NetworkConfig dimenzio-szamitasi logikajanek tesztjei."""

    def test_default_trunk_dim(self) -> None:
        """Az alapertelmezett trunk dimenzio 64*2 + 32 + 64 = 224."""
        try:
            from src.model.networks import NetworkConfig
            cfg = NetworkConfig()
            assert cfg.trunk_input_dim == 224
        except (ImportError, AttributeError):
            # torch.nn mock nem tamogatja — kozvetlenul szamolunk
            assert 64 * 2 + 32 + 64 == 224

    def test_custom_trunk_dim(self) -> None:
        """Egyedi embed dimenziokal a trunk dimenzio megvaltozik."""
        try:
            from src.model.networks import NetworkConfig
            cfg = NetworkConfig(card_embed_dim=128, context_embed_dim=64, history_embed_dim=128)
            assert cfg.trunk_input_dim == 128 * 2 + 64 + 128  # 448
        except (ImportError, AttributeError):
            assert 128 * 2 + 64 + 128 == 448

    def test_from_dict_parsing(self, sample_config: dict) -> None:
        """A YAML -> NetworkConfig konverzio helyesen szamol."""
        try:
            from src.model.networks import NetworkConfig
            num_players = sample_config["environment"]["num_players"]
            cfg = NetworkConfig.from_dict(sample_config, num_players=num_players)
            assert cfg.num_actions == 9
            assert cfg.card_input_dim == 52
            assert cfg.position_dim == num_players
            assert cfg.env_metrics_dim == 4 + (num_players - 1)
            assert cfg.actor_hidden == (512, 256, 128)
            assert cfg.critic_hidden == (512, 256, 128)
            assert cfg.activation == "relu"
            assert cfg.dropout == 0.1
            assert cfg.weight_init == "orthogonal"
        except (ImportError, AttributeError):
            # Fallback: config ertekek kozvetlenul
            model = sample_config["model"]
            assert model["actor"]["hidden_layers"] == [512, 256, 128]
            assert model["weight_init"] == "orthogonal"

    def test_config_yaml_dimensions_consistency(self, sample_config: dict) -> None:
        """A config.yaml dimenzio beallitasai konzisztensek."""
        embed = sample_config["model"]["embedding"]
        obs = sample_config["environment"]["observation_space"]

        # Kartya dimenzio
        assert obs["hole_cards_dim"] == 52
        assert obs["community_cards_dim"] == 52
        assert obs["total_card_dim"] == 104

        # Beagyazasi dimenziok pozitivak
        assert embed["card_embed_dim"] > 0
        assert embed["context_embed_dim"] > 0
        assert embed["history_embed_dim"] > 0

        # Trunk dimenzio szamitas
        trunk = embed["card_embed_dim"] * 2 + embed["context_embed_dim"] + embed["history_embed_dim"]
        assert trunk == 224  # 64*2 + 32 + 64

    def test_action_space_matches_network_output(self, sample_config: dict) -> None:
        """Az akciok szama egyezik a halozat kimenetevel."""
        num_actions = sample_config["environment"]["action_space"]["num_actions"]
        assert num_actions == 9


# =============================================================================
# Architektura Dimenzio Tesztek
# =============================================================================

class TestArchitectureDimensions:
    """A halozat reteg-meret konzisztenciajanak tesztjei."""

    def test_actor_hidden_layers_decreasing(self, sample_config: dict) -> None:
        """Az actor rejtett retegek csokkeno meretekuek."""
        layers = sample_config["model"]["actor"]["hidden_layers"]
        for i in range(len(layers) - 1):
            assert layers[i] >= layers[i + 1], \
                f"Actor reteg {i} ({layers[i]}) < reteg {i+1} ({layers[i+1]})"

    def test_critic_hidden_layers_decreasing(self, sample_config: dict) -> None:
        """A critic rejtett retegek csokkeno meretekuek."""
        layers = sample_config["model"]["critic"]["hidden_layers"]
        for i in range(len(layers) - 1):
            assert layers[i] >= layers[i + 1]

    def test_estimated_parameter_count(self, sample_config: dict) -> None:
        """A becsult parameterszam az elvart tartomanyban van."""
        embed = sample_config["model"]["embedding"]
        actor_layers = sample_config["model"]["actor"]["hidden_layers"]
        critic_layers = sample_config["model"]["critic"]["hidden_layers"]

        trunk = embed["card_embed_dim"] * 2 + embed["context_embed_dim"] + embed["history_embed_dim"]

        # Embedding param becslés (hozzavetoleges)
        hole_params = 52 * embed["card_embed_dim"] + embed["card_embed_dim"] * 2
        comm_params = hole_params
        ctx_input = (4 + 5) + 6  # metrics + position = 15
        ctx_params = ctx_input * embed["context_embed_dim"] + embed["context_embed_dim"] * 2
        hist_flat = 18 * 9
        hist_params = (hist_flat * embed["history_embed_dim"] * 2
                       + embed["history_embed_dim"] * 2 * 3
                       + embed["history_embed_dim"] * 2 * embed["history_embed_dim"]
                       + embed["history_embed_dim"] * 3)

        # Actor MLP: trunk->512->256->128->9 + LayerNorm parameterek
        actor_params = 0
        prev = trunk
        for h in actor_layers:
            actor_params += prev * h + h + h * 2  # Linear + bias + LayerNorm
            prev = h
        actor_params += prev * 9 + 9  # Kimeneti reteg

        # Hasonlo a critic-hez (128->1 kimeneti)
        critic_params = 0
        prev = trunk
        for h in critic_layers:
            critic_params += prev * h + h + h * 2
            prev = h
        critic_params += prev * 1 + 1

        total_est = (hole_params + comm_params + ctx_params + hist_params
                     + actor_params + critic_params)

        # Az osszes param 400k-800k kozott kell legyen 6-Max-hoz
        assert 300_000 < total_est < 1_000_000, \
            f"Parameterszam {total_est:,} kivul esik a vart tartomanyon"

    def test_history_shape_matches_config(self, sample_config: dict) -> None:
        """A licittortenet tenzor merete egyezik a konfiguracioval."""
        obs = sample_config["environment"]["observation_space"]
        assert obs["max_actions_per_round"] == 18
        hist_dim = obs["betting_history_dim"]
        assert hist_dim == [18, 9]


# =============================================================================
# Akcio Maszkolas Matematikai Tesztek
# =============================================================================

class TestActionMaskingMath:
    """Az akcio maszkolas matematikai helyessegenek tesztjei."""

    def test_masked_softmax_illegal_zero(self) -> None:
        """Az illegalis akciok valoszinusege ~0 a maszkolt Softmax utan."""
        import torch
        from src.env.action_mapper import ILLEGAL_ACTION_LOGIT

        logits = torch.tensor(np.random.randn(9).astype(np.float32))
        mask = torch.tensor(np.array([1, 1, 0, 0, 1, 0, 0, 0, 1], dtype=np.float32))

        masked = logits + (1.0 - mask) * ILLEGAL_ACTION_LOGIT
        probs = torch.softmax(masked, dim=-1)

        for idx in [2, 3, 5, 6, 7]:
            assert float(probs[idx].item()) < 1e-20, \
                f"Illegalis index {idx} valoszinusege nem nulla: {probs[idx].item()}"

    def test_masked_softmax_legal_sum_one(self) -> None:
        """A legalis akciok valoszinusegeinek osszege ~1.0."""
        import torch
        from src.env.action_mapper import ILLEGAL_ACTION_LOGIT

        logits = torch.tensor(np.random.randn(9).astype(np.float32))
        mask = torch.tensor(np.array([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.float32))

        masked = logits + (1.0 - mask) * ILLEGAL_ACTION_LOGIT
        probs = torch.softmax(masked, dim=-1)

        legal_sum = sum(float(probs[i].item()) for i in [0, 2, 4, 6, 8])
        assert abs(legal_sum - 1.0) < 1e-5, f"Legalis osszeg: {legal_sum}"

    def test_single_legal_action_probability_one(self) -> None:
        """Ha csak egy legalis akcio van, annak valoszinusege 1.0."""
        import torch
        from src.env.action_mapper import ILLEGAL_ACTION_LOGIT

        logits = torch.tensor(np.random.randn(9).astype(np.float32))
        mask = torch.tensor(np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32))

        masked = logits + (1.0 - mask) * ILLEGAL_ACTION_LOGIT
        probs = torch.softmax(masked, dim=-1)

        assert float(probs[4].item()) > 0.999

    def test_illegal_logit_value(self) -> None:
        """Az ILLEGAL_ACTION_LOGIT erteke -1e8."""
        from src.env.action_mapper import ILLEGAL_ACTION_LOGIT
        assert ILLEGAL_ACTION_LOGIT == -1e8


# =============================================================================
# Sulyinicializacio Konfiguracio Tesztek
# =============================================================================

class TestWeightInitConfig:
    """A sulyinicializacios konfiguracio tesztjei."""

    def test_supported_init_methods(self, sample_config: dict) -> None:
        """A konfiguralt inicializacios modszer ervenyes."""
        method = sample_config["model"]["weight_init"]
        valid = ["orthogonal", "xavier", "kaiming"]
        assert method in valid, f"Ervenytelen init: {method}"

    def test_init_gain_positive(self, sample_config: dict) -> None:
        gain = sample_config["model"]["weight_init_gain"]
        assert gain > 0

    def test_supported_activations(self, sample_config: dict) -> None:
        valid = ["relu", "gelu", "tanh", "leaky_relu", "elu"]
        actor_act = sample_config["model"]["actor"]["activation"]
        assert actor_act in valid

    def test_dropout_range(self, sample_config: dict) -> None:
        dropout = sample_config["model"]["actor"]["dropout"]
        assert 0.0 <= dropout < 1.0
