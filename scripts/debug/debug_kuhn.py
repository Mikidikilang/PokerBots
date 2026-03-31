#!/usr/bin/env python3
"""Debug script to test CFR convergence."""

import numpy as np
from test_kuhn_nash_convergence import KuhnCFRSolver

solver = KuhnCFRSolver()

# Run iterations with checkpoints
for iterations in [100, 500, 1000, 5000, 10000]:
    solver = KuhnCFRSolver()  # Fresh solver
    
    for it in range(iterations):
        solver.run_iteration()
    
    print(f"\n=== After {iterations} iterations ===")
    
    for card_idx, card_name in enumerate(["Jack", "Queen", "King"]):
        infoset_id = f"P0_{card_name}_root"
        if infoset_id in solver.infosets:
            avg_strategy = solver.get_average_strategy(infoset_id)
            iset = solver.infosets[infoset_id]
            current_strategy = solver._compute_strategy(iset)
            print(f"{card_name:6s}: Avg BET {avg_strategy[1]*100:6.2f}%   Current BET {current_strategy[1]*100:6.2f}%   Regrets={iset['cumulative_regrets']}")

