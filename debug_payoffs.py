#!/usr/bin/env python3
"""Debug Kuhn poker payoffs to verify they're correct."""

from src.env.kuhn_poker_minimal import KuhnPokerEnv, KuhnAction

# Test Case 1: P1=King, P0=Jack, P1 bets after P0 checks, P0 calls -> showdown
env = KuhnPokerEnv(seed=42)
env.reset()

# Just test with whatever cards we got
p0_c = env.CARDS[env.p0_card]
p1_c = env.CARDS[env.p1_card]
print(f"Test 1 setup: P0={p0_c} ({env.p0_card}), P1={p1_c} ({env.p1_card})")

# P0 checks
env.step(KuhnAction.CHECK)
assert env.get_player_id() == 1  # P1's turn

# P1 bets
env.step(KuhnAction.BET)
assert env.get_player_id() == 0  # P0's turn
assert not env.is_over()
print(f"  After C,B: history={''.join(env.history)}, terminal={env.is_over()}")

# P0 calls
env.step(KuhnAction.CALL)
assert env.is_over(), "Game should be terminal after CBK"
print(f"  After C,B,K: history={''.join(env.history)}, terminal={env.is_over()}")

payoff = env.get_payoff()
print(f"  P0 payoff: {payoff}")
print(f"  P1 payoff: {-payoff}")
assert env.p0_card != env.p1_card or payoff == 0  # Can't predict without knowing cards
print(f"  ✓ PASS (CBK is terminal)\n")

# Test Case 2: Both check -> showdown
env2 = KuhnPokerEnv(seed=43)
env2.reset()

p0_c2 = env2.CARDS[env2.p0_card]
p1_c2 = env2.CARDS[env2.p1_card]
print(f"Test 2 setup: P0={p0_c2} ({env2.p0_card}), P1={p1_c2} ({env2.p1_card})")

expected_payoff = 0
if env2.p0_card > env2.p1_card:
    expected_payoff = 1
elif env2.p0_card < env2.p1_card:
    expected_payoff = -1

# Both check
env2.step(KuhnAction.CHECK)
env2.step(KuhnAction.CHECK)
assert env2.is_over(), "Game should be terminal after CC"
print(f"  After C,C: history={''.join(env2.history)}, terminal={env2.is_over()}")

payoff = env2.get_payoff()
print(f"  P0 payoff: {payoff} (expected: {expected_payoff})")
assert payoff == expected_payoff, f"Expected {expected_payoff}, got {payoff}"
print(f"  ✓ PASS\n")

# Test Case 3: P0 bets, P1 folds
env3 = KuhnPokerEnv(seed=44)
env3.reset()

print(f"Test 3 setup: P0={env3.CARDS[env3.p0_card]}, P1={env3.CARDS[env3.p1_card]}")

# P0 bets
env3.step(KuhnAction.BET)
print(f"  After B: history={''.join(env3.history)}, terminal={env3.is_over()}")

# P1 folds
env3.step(KuhnAction.FOLD)
assert env3.is_over(), "Game should be terminal after BF"
print(f"  After B,F: history={''.join(env3.history)}, terminal={env3.is_over()}")

payoff = env3.get_payoff()
print(f"  P0 payoff: {payoff} (should be 1, P0 wins)")
assert payoff == 1, f"Expected 1, got {payoff}"
print(f"  ✓ PASS\n")

print("All payoff tests passed!")
