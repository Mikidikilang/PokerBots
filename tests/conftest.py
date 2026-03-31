"""
Kozos Pytest Fixture-ok es Mock Infrastruktura (conftest.py).

Ez a fajl biztositja a megosztott teszt fixture-oket es a torch mock
rendszert az osszes tesztmodul szamara. A mock lehetove teszi a tesztek
futtatast torch telepites nelkul is (CI/CD kompatibilitas).
"""

from __future__ import annotations

import os
import sys
import random
import tempfile
from pathlib import Path
from typing import Any, Generator

import numpy as np
import pytest
import yaml

# =============================================================================
# Torch Mock Infrastruktura
# =============================================================================
# A mock CSAK akkor aktivalodik, ha a valodi torch nem elerheto.
# Produkcios kornyezetben a valodi torch-ot hasznalja.
# =============================================================================

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    import types

    torch_mock = types.ModuleType("torch")

    class FakeTensor:
        """Minimalis Tensor mock a tesztekhez."""

        def __init__(self, data: Any = None, dtype: Any = None) -> None:
            if isinstance(data, FakeTensor):
                self._data = data._data.copy()
            elif isinstance(data, np.ndarray):
                self._data = data.astype(np.float32)
            elif isinstance(data, (list, tuple)):
                self._data = np.array(data, dtype=np.float32)
            elif isinstance(data, (int, float)):
                self._data = np.array([data], dtype=np.float32)
            elif data is None:
                self._data = np.array([0.0], dtype=np.float32)
            else:
                self._data = np.atleast_1d(np.array(data, dtype=np.float32))
            self.shape = self._data.shape
            self.dtype = dtype or "float32"

        def __getitem__(self, k: Any) -> FakeTensor:
            r = self._data[k]
            return FakeTensor(np.atleast_1d(r) if not isinstance(r, np.ndarray) else r)

        def __setitem__(self, k: Any, v: Any) -> None:
            self._data[k] = v

        def item(self) -> float:
            return float(self._data.flat[0])

        def sum(self) -> FakeTensor:
            return FakeTensor(np.array([self._data.sum()]))

        def min(self) -> FakeTensor:
            return FakeTensor(np.array([self._data.min()]))

        def max(self) -> FakeTensor:
            return FakeTensor(np.array([self._data.max()]))

        def mean(self) -> FakeTensor:
            return FakeTensor(np.array([self._data.mean()]))

        def std(self) -> FakeTensor:
            return FakeTensor(np.array([max(self._data.std(), 1e-8)]))

        def flatten(self) -> FakeTensor:
            return FakeTensor(self._data.flatten())

        def reshape(self, *s: int) -> FakeTensor:
            return FakeTensor(self._data.reshape(*s))

        def unsqueeze(self, d: int) -> FakeTensor:
            return FakeTensor(np.expand_dims(self._data, d))

        def squeeze(self, d: int | None = None) -> FakeTensor:
            return FakeTensor(self._data.squeeze() if d is None else np.squeeze(self._data, d))

        def numel(self) -> int:
            return self._data.size

        def bool(self) -> np.ndarray:
            return self._data.astype(bool)

        def any(self) -> bool:
            return bool(self._data.any())

        def dim(self) -> int:
            return len(self.shape)

        def tolist(self) -> list:
            return self._data.tolist()

        def detach(self) -> FakeTensor:
            return self

        def long(self) -> FakeTensor:
            return self

        def to(self, *a: Any, **kw: Any) -> FakeTensor:
            return self

        def clone(self) -> FakeTensor:
            return FakeTensor(self._data.copy())

        def view(self, *shape: int) -> FakeTensor:
            return FakeTensor(self._data.reshape(shape))

        def float(self) -> FakeTensor:
            return FakeTensor(self._data.astype(np.float32))

        def contiguous(self) -> FakeTensor:
            return FakeTensor(np.ascontiguousarray(self._data))

        def requires_grad_(self, requires_grad: bool = True) -> FakeTensor:
            return self

        @property
        def grad(self) -> None:
            return None

        def norm(self) -> FakeTensor:
            return FakeTensor(np.array([np.linalg.norm(self._data)]))

        def __add__(self, o: Any) -> FakeTensor:
            return FakeTensor(self._data + (o._data if isinstance(o, FakeTensor) else o))

        def __radd__(self, o: Any) -> FakeTensor:
            return FakeTensor((o._data if isinstance(o, FakeTensor) else o) + self._data)

        def __sub__(self, o: Any) -> FakeTensor:
            return FakeTensor(self._data - (o._data if isinstance(o, FakeTensor) else o))

        def __rsub__(self, o: Any) -> FakeTensor:
            return FakeTensor((o._data if isinstance(o, FakeTensor) else o) - self._data)

        def __mul__(self, o: Any) -> FakeTensor:
            return FakeTensor(self._data * (o._data if isinstance(o, FakeTensor) else o))

        def __rmul__(self, o: Any) -> FakeTensor:
            return FakeTensor((o._data if isinstance(o, FakeTensor) else o) * self._data)

        def __truediv__(self, o: Any) -> FakeTensor:
            return FakeTensor(self._data / (o._data if isinstance(o, FakeTensor) else o))

        def __eq__(self, o: Any) -> FakeTensor:
            return FakeTensor((self._data == (o._data if isinstance(o, FakeTensor) else o)).astype(np.float32))

        def __repr__(self) -> str:
            return f"FakeTensor(shape={self.shape})"

    torch_mock.Tensor = FakeTensor
    torch_mock.float32 = "float32"
    torch_mock.zeros = lambda *a, dtype=None: FakeTensor(np.zeros(a[0] if len(a) == 1 and isinstance(a[0], (list, tuple)) else (a[0],) if len(a) == 1 else a))
    torch_mock.tensor = lambda d, dtype=None: FakeTensor(d)
    
    def _fake_cat(tensors: list, dim: int = 0) -> FakeTensor:
        """Concatenate tensors along specified dimension (respects dim arg)."""
        arrays = [t._data if hasattr(t, '_data') else np.array(t) for t in tensors]
        return FakeTensor(np.concatenate(arrays, axis=dim))
    torch_mock.cat = _fake_cat
    
    torch_mock.stack = lambda t, dim=0: FakeTensor(np.stack([x._data for x in t], axis=dim))
    torch_mock.randn = lambda *a: FakeTensor(np.random.randn(*a))
    torch_mock.softmax = lambda t, dim=-1: FakeTensor(np.exp(t._data - t._data.max()) / np.exp(t._data - t._data.max()).sum())
    torch_mock.device = lambda x: x
    torch_mock.no_grad = lambda: types.SimpleNamespace(__enter__=lambda s: None, __exit__=lambda s, *a: None)
    torch_mock.save = lambda o, p: __import__("pickle").dump(o, open(str(p), "wb"))
    torch_mock.load = lambda p, **kw: __import__("pickle").load(open(str(p), "rb")) if os.path.exists(str(p)) else {}
    torch_mock.get_rng_state = lambda: FakeTensor(np.array([42]))
    torch_mock.set_rng_state = lambda s: None
    torch_mock.manual_seed = lambda s: None
    
    # Phase 3-19: Add torch.where and torch.finfo mocks for gradient health checks
    torch_mock.where = lambda condition, x, y: FakeTensor(np.where(condition._data if hasattr(condition, '_data') else condition, x._data if hasattr(x, '_data') else x, y._data if hasattr(y, '_data') else y))
    
    class _FakeFinfo:
        """Mock for torch.finfo to provide dtype properties."""
        def __init__(self, dtype: Any) -> None:
            self.max = 3.4028235e38  # float32 max
            self.min = -3.4028235e38  # float32 min
            self.tiny = 1.1754944e-38  # float32 tiny
    torch_mock.finfo = _FakeFinfo

    class _FakeGenerator:
        _state = FakeTensor(np.array([0]))
        def get_state(self) -> FakeTensor: return self._state
        def set_state(self, s: Any) -> None: self._state = s
        def manual_seed(self, s: int) -> None: pass
    torch_mock.Generator = _FakeGenerator

    # CUDA mock
    cuda_mod = types.ModuleType("torch.cuda")
    cuda_mod.is_available = lambda: False
    torch_mock.cuda = cuda_mod

    # Distributions mock
    dist_mod = types.ModuleType("torch.distributions")
    class _FakeCategorical:
        def __init__(self, probs: Any = None, logits: Any = None) -> None:
            """Initialize categorical distribution from logits or probs with numerically stable softmax."""
            if logits is not None:
                # Extract numpy array from FakeTensor or convert input
                data = logits._data if hasattr(logits, '_data') else np.array(logits, dtype=np.float32)
                # Numerically stable softmax: subtract max before exp
                data_shifted = data - data.max()
                exp_data = np.exp(data_shifted)
                self._p = exp_data / exp_data.sum()
            elif probs is not None:
                # Use provided probabilities (normalize if needed)
                data = probs._data if hasattr(probs, '_data') else np.array(probs, dtype=np.float32)
                self._p = data / data.sum()
            else:
                # Default uniform distribution over 9 actions
                self._p = np.ones(9, dtype=np.float32) / 9
            self.probs = FakeTensor(self._p)
            self.logits = logits if logits is not None else probs

        def sample(self) -> FakeTensor:
            """Sample action from the categorical distribution."""
            action = np.random.choice(len(self._p), p=self._p)
            return FakeTensor(np.array([float(action)]))

        def log_prob(self, v: Any) -> FakeTensor:
            """Compute log probability of action according to current distribution."""
            action_idx = int(v.item()) if hasattr(v, 'item') else int(v)
            if 0 <= action_idx < len(self._p):
                log_p = np.log(np.maximum(self._p[action_idx], 1e-8))
            else:
                log_p = np.log(1e-8)
            return FakeTensor(np.array([log_p]))

        def entropy(self) -> FakeTensor:
            """Compute Shannon entropy of the categorical distribution."""
            # H = -sum(p * log(p))
            ent = -np.sum(self._p * np.log(np.maximum(self._p, 1e-8)))
            return FakeTensor(np.array([ent]))
    dist_mod.Categorical = _FakeCategorical
    torch_mock.distributions = dist_mod

    # nn mock
    nn_mod = types.ModuleType("torch.nn")
    nn_mod.Module = type("Module", (), {"parameters": lambda s: [], "train": lambda s: None, "eval": lambda s: None})
    torch_mock.nn = nn_mod

    # optim mock
    optim_mod = types.ModuleType("torch.optim")
    class _FakeAdam:
        def __init__(self, params: Any = None, lr: float = 0.001, eps: float = 1e-8) -> None:
            self.param_groups = [{"lr": lr}]
        def zero_grad(self) -> None: pass
        def step(self) -> None: pass
        def state_dict(self) -> dict: return {}
        def load_state_dict(self, d: dict) -> None: pass
    optim_mod.Adam = _FakeAdam
    torch_mock.optim = optim_mod

    # Register mocks
    sys.modules["torch"] = torch_mock
    sys.modules["torch.cuda"] = cuda_mod
    sys.modules["torch.nn"] = nn_mod
    sys.modules["torch.nn.functional"] = types.ModuleType("torch.nn.functional")
    sys.modules["torch.distributions"] = dist_mod
    sys.modules["torch.optim"] = optim_mod

    torch = torch_mock


# =============================================================================
# Pytest Fixture-ok
# =============================================================================

@pytest.fixture
def sample_config() -> dict[str, Any]:
    """Betolti a config.yaml-t teszteleshez."""
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_raw_state() -> dict[str, Any]:
    """Egy tipikus nyers jatekallapot a features.py teszteleshez."""
    return {
        "hand": ["SA", "HK"],
        "public_cards": ["CT", "DJ", "SQ"],
        "pot": 150.0,
        "my_chips": 1800.0,
        "opponent_chips": [2000.0, 1500.0, 1000.0, 800.0, 2200.0],
        "big_blind": 10.0,
        "amount_to_call": 50.0,
        "min_raise": 100.0,
        "position": 5,
        "betting_history": [
            {"action": 3, "amount": 50, "player": 0},
            {"action": 1, "amount": 50, "player": 1},
        ],
        "legal_actions": [0, 1, 3, 4, 5, 8],
    }


@pytest.fixture
def sample_preflop_state() -> dict[str, Any]:
    """Pre-flop allapot (ures board)."""
    return {
        "hand": ["HA", "DA"],
        "public_cards": [],
        "pot": 15.0,
        "my_chips": 2000.0,
        "opponent_chips": [2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
        "big_blind": 10.0,
        "amount_to_call": 10.0,
        "min_raise": 20.0,
        "position": 0,
        "betting_history": [],
        "legal_actions": [0, 1, 2, 3, 4, 5, 6, 7, 8],
    }


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Ideiglenes konyvtar a fajl-muveletekhez."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
