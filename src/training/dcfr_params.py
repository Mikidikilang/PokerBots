"""
Phase 3: Discounted CFR (DCFR) Implementation

[PHASE 3] Discounted Counterfactual Regret Minimization with per-sign discount factors.

References:
    - Brown & Sandholm (2019): "Solving Imperfect-Information Games via Discounted 
      Regret Minimization" (optimal parameters: α=1.5, β=0, γ=2)
    - Optimal discount formula:
      R^new(a) = T / (T + 1) * R^old(a) + r(a)    [positive regrets]
      R^new(a) = T / (T + 1) * R^old(a) + r(a)    [negative regrets]
    
    With per-sign discounts:
      - If R^old(a) > 0: discount_pos = (T / (T + γ))^α
      - If R^old(a) ≤ 0: discount_pos = (T / (T + γ))^β
    
    Standard parameters (Brown & Sandholm 2019):
      α = 1.5 (apply stronger discount to positive regrets)
      β = 0   (no discount to negative regrets)
      γ = 2   (discount base adjustment)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DCFRParameters:
    """Discounted CFR hyperparameters (Brown & Sandholm 2019)."""
    
    alpha: float = 1.5      # Exponent for positive regret discounting
    beta: float = 0.0       # Exponent for negative regret discounting
    gamma: float = 2.0      # Discount base shift
    
    # Legacy RM+ compatibility
    use_legacy_rm_plus: bool = False
    legacy_discount_factor: float = 3.0


def compute_dcfr_discount(
    iteration: int,
    regret_old: float,
    params: DCFRParameters,
) -> float:
    """
    Compute adaptive discount factor using Brown & Sandholm DCFR formula.
    
    Formula:
        If regret_old > 0 (positive):   discount = (t / (t + γ))^α
        If regret_old ≤ 0 (negative):   discount = (t / (t + γ))^β
    
    Where t = iteration (1-indexed for numerical stability).
    
    Args:
        iteration: CFR iteration number (0-indexed from algo, convert to 1-indexed)
        regret_old: Previous cumulative regret for this action
        params: DCFR parameters
    
    Returns:
        Discount factor in (0, 1]. Higher = stronger emphasis on recent regrets.
    """
    # Convert to 1-indexed iteration for formula
    t = float(iteration + 1)
    
    if regret_old > 1e-8:
        # Positive regret: apply alpha discount
        exponent = params.alpha
    else:
        # Negative/zero regret: apply beta discount
        exponent = params.beta
    
    # Base discount calculation
    # (t / (t + γ))^α
    base_ratio = t / (t + params.gamma)
    discount = math.pow(base_ratio, exponent)
    
    # Clamp to [0, 1] for numerical safety
    discount = max(0.0, min(1.0, discount))
    
    return discount


def apply_dcfr_update(
    regret_old: float,
    regret_new: float,
    iteration: int,
    params: DCFRParameters,
) -> float:
    """
    Apply DCFR update with adaptive per-sign discounting.
    
    R^new(a) = discount * R^old(a) + r^new(a)
    
    Args:
        regret_old: Cumulative regret at start of iteration
        regret_new: Counterfactual regret in this iteration
        iteration: Current iteration number
        params: DCFR parameters
    
    Returns:
        Updated cumulative regret
    """
    discount = compute_dcfr_discount(iteration, regret_old, params)
    regret_updated = discount * regret_old + regret_new
    
    return regret_updated


# ============================================================================
# Compatibility: Legacy RM+ Mode
# ============================================================================

def compute_legacy_rm_plus_discount(discount_factor: float = 3.0) -> float:
    """
    Legacy RM+ discount (constant, non-adaptive).
    
    R^new(a) = discount_factor * R^old(a) + r(a)
    
    Standard value: 3.0 or 4.0 (higher = more emphasis on recent).
    
    Args:
        discount_factor: Constant multiplier
    
    Returns:
        discount_factor (no iteration dependence)
    """
    return discount_factor


# ============================================================================
# Verification
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    params = DCFRParameters(alpha=1.5, beta=0.0, gamma=2.0)
    
    print("DCFR Discount Evolution (α=1.5, β=0, γ=2):")
    print("Iteration | Positive Regret | Negative Regret")
    print("-" * 50)
    
    for iteration in [0, 1, 10, 100, 1000]:
        disc_pos = compute_dcfr_discount(iteration, regret_old=0.5, params=params)
        disc_neg = compute_dcfr_discount(iteration, regret_old=-0.5, params=params)
        print(f"{iteration:9d} | {disc_pos:15.6f} | {disc_neg:15.6f}")
    
    print("\nDCFR Update Example:")
    regret = 0.0
    for i in range(5):
        new_regret_value = 0.1  # Constant inbound regret
        regret = apply_dcfr_update(regret, new_regret_value, i, params)
        print(f"  Iteration {i}: regret = {regret:.6f}")
