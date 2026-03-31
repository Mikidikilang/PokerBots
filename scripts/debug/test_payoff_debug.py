#!/usr/bin/env python3
"""Debug payoff computation for small example."""

from test_kuhn_nash_convergence import KuhnCFRSolver

solver = KuhnCFRSolver()

# Just one iteration with explicit output
print("=" * 70)
print("KUHN POKER PAYOFF DEBUG")
print("=" * 70)

# Test case manually
# P0 has Jack (0), P1 has Queen (1)
# History: ("check", "check") - showdown
# P0 loses, so payoff_p0 = -1
#
# When updating P0's regrets:
#   value_for_p0 = -1 (P0 loses)
#
# When updating P1's regrets:
#   value_for_p1 = -(-1) = +1 (P1 wins+)

p0_card, p1_card = 0, 1  # Jack vs Queen
history = ("check", "check")

# Compute payoff (Jack vs Queen, showdown)
is_p0_wins = p0_card > p1_card  # 0 > 1? No, so P0 loses
payoff_p0 = -1 if not is_p0_wins else 1

print(f"\nScenario: P0 has {KuhnCFRSolver.CARDS[p0_card]}, P1 has {KuhnCFRSolver.CARDS[p1_card]}")
print(f"History: {history}")
print(f"P0 wins card comparison: {is_p0_wins}")
print(f"Payoff from P0's perspective: {payoff_p0}")
print(f"Payoff from P1's perspective (negated): {-payoff_p0}")

print("\nWhen updating P0's regrets:")
print(f"  Terminal value returned: {payoff_p0}")
print(f"  Regrets computed from P0's perspective")

print("\nWhen updating P1's regrets:")
print(f"  Terminal value returned: {-payoff_p0}")
print(f"  Regrets computed from P1's perspective (NEGATED)")

print("\n" + "=" * 70)
print("Running actual iterative CFR to verify:")
print("=" * 70)

solver.run_iteration()

# Check what regrets were accumulated for Jack against Queen
# We want to know: does Jack prefer to CHECK or BET?

print("\nInformations about strategy development:")
for iterations_target in [1, 10, 100, 1000]:
    solver2 = KuhnCFRSolver()
    for i in range(iterations_target):
        solver2.run_iteration()
    
    jack_iid = "P0_Jack_root"
    if jack_iid in solver2.infosets:
        iset = solver2.infosets[jack_iid]
        current = solver2._compute_strategy(iset)
        avg = solver2.get_average_strategy(jack_iid)
        print(f"\nAfter {iterations_target} iterations (Jack):")
        print(f"  Cumulative regrets: {iset['cumulative_regrets']}")
        print(f"  Current strategy BET: {current[1]*100:.2f}%")
        print(f"  Average strategy BET: {avg[1]*100:.2f}%")
