#!/usr/bin/env python3
"""Simple test of regret convergence behavior."""

from test_kuhn_nash_convergence import KuhnCFRSolver

solver = KuhnCFRSolver()

for i in range(10000):
    solver.run_iteration()
    if (i + 1) in [100, 1000, 5000, 10000]:
        print(f"\nAfter {i+1} iterations:")
        for card_idx, card_name in enumerate(['Jack', 'Queen', 'King']):
            iid = f'P0_{card_name}_root'
            if iid in solver.infosets:
                iset = solver.infosets[iid]
                current_strat = solver._compute_strategy(iset)
                avg_strategy = solver.get_average_strategy(iid)
                print(f"  {card_name}: BET current={current_strat[1]*100:6.2f}%, avg={avg_strategy[1]*100:6.2f}%")
                print(f"           regrets={iset['cumulative_regrets']}")

