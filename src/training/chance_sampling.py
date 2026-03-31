"""
Chance Node Sampling for CFR (Phase 3)

[PHASE 3] Proper card dealing from deck distribution, treating hero's cards 
as private information during game tree traversal.

Key Principles:
    1. **Deck Distribution**: Deal from actual 52-card deck, not fixed hand
    2. **Private Information**: Hero's hole cards are unknown to opponent
    3. **Public Information**: Board cards are known to both players
    4. **Importance Weighting**: Each card combo has different probability
    
Problem with Previous Approach:
    - Hero's specific cards (e.g., As,Ks) were fixed at traversal start
    - Opponent knew hero's cards (leaked information)
    - Unrealistic: humans don't see opponent hole cards during play
    
Solution:
    - During traversal: treat hero's hole cards as private info (opponent samples)
    - Use importance weighting: w(cards) = 1 / P(cards | observed info)
    - Support counterfactual sampling: compute values vs different opponent hands

References:
    - Kuhn (1950): "A Simplified Two-Person Poker"
    - Koller & Megiddo (1992): "The Complexity of Two-Player Zero-Sum Games"
    - Brown & Sandholm (2019): "Solving Imperfect-Information Games"
"""

from __future__ import annotations

import itertools
import logging
import random
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Standard 52-card deck
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['s', 'h', 'd', 'c']
DECK = [f"{rank}{suit}" for rank in RANKS for suit in SUITS]


@dataclass
class DeckState:
    """Current state of deck during card dealing."""
    
    remaining_cards: list[str]
    """Cards not yet dealt"""
    
    dealt_cards: dict[str, list[str]] = None
    """dealt_cards[player_id] = list of cards dealt to that player"""
    
    board_cards: list[str] = None
    """Community cards (known to all players)"""
    
    def __post_init__(self):
        if self.dealt_cards is None:
            self.dealt_cards = {'hero': [], 'opponent': []}
        if self.board_cards is None:
            self.board_cards = []
    
    @staticmethod
    def create_fresh_deck() -> DeckState:
        """Create fresh 52-card deck."""
        return DeckState(remaining_cards=DECK.copy())
    
    def deal_hole_cards(self, player: str, num_cards: int = 2) -> list[str]:
        """
        Deal specified number of hole cards to a player.
        
        Args:
            player: 'hero' or 'opponent'
            num_cards: Usually 2 (Texas Hold'em)
        
        Returns:
            list of dealt cards
        """
        if len(self.remaining_cards) < num_cards:
            raise ValueError(f"Not enough cards: need {num_cards}, have {len(self.remaining_cards)}")
        
        dealt = random.sample(self.remaining_cards, num_cards)
        
        for card in dealt:
            self.remaining_cards.remove(card)
        
        self.dealt_cards[player].extend(dealt)
        return dealt
    
    def deal_board_cards(self, num_cards: int, stage: str = '') -> list[str]:
        """
        Deal community cards (flop=3, turn=1, river=1).
        
        Args:
            num_cards: Number of cards to deal
            stage: 'flop', 'turn', or 'river' (for logging)
        
        Returns:
            list of newly dealt cards
        """
        if len(self.remaining_cards) < num_cards:
            raise ValueError(f"Not enough cards for {stage}: need {num_cards}")
        
        dealt = random.sample(self.remaining_cards, num_cards)
        
        for card in dealt:
            self.remaining_cards.remove(card)
        
        self.board_cards.extend(dealt)
        return dealt
    
    def get_remaining_count(self) -> int:
        """Number of undealt cards."""
        return len(self.remaining_cards)


@dataclass
class CardInfo:
    """Information about cards known to a player."""
    
    player_id: str
    """'hero' or 'opponent'"""
    
    known_cards: dict[str, list[str]]
    """known_cards['hero'] = hero's hole cards (if we are hero, known; if opponent, unknown during traversal)"""
    
    board_cards: list[str]
    """board_cards = public community cards"""
    
    def get_visible_cards(self) -> set[str]:
        """Cards visible to this player."""
        visible = set(self.board_cards)
        
        # If we're the player, we know our own hole cards
        if self.player_id == 'hero' and 'hero' in self.known_cards:
            visible.update(self.known_cards['hero'])
        
        return visible
    
    def get_remaining_cards(self) -> list[str]:
        """Cards potentially in opponent's hand."""
        visible = self.get_visible_cards()
        remaining = [card for card in DECK if card not in visible]
        return remaining


def enumerate_opponent_hands(
    known_cards: set[str],
    deck: list[str] = DECK,
    opponent_hand_size: int = 2,
) -> list[Tuple[str, str]]:
    """
    Enumerate all possible opponent hole card combinations.
    
    Given observed cards (hero's hole + board), what hole card combos
    could the opponent have?
    
    Args:
        known_cards: Cards visible to us (hero hole + board)
        deck: Full deck to sample from
        opponent_hand_size: Usually 2
    
    Returns:
        list of tuples (card1, card2) for opponent's possible hands
    """
    available = [c for c in deck if c not in known_cards]
    opponent_hands = list(itertools.combinations(available, opponent_hand_size))
    return opponent_hands


def sample_opponent_hands(
    known_cards: set[str],
    num_samples: int,
    deck: list[str] = DECK,
    opponent_hand_size: int = 2,
) -> list[Tuple[str, str]]:
    """
    Sample opponent hands for Monte Carlo approximation.
    
    For large hand spaces, enumerating all combinations is expensive.
    Instead, sample uniformly random opponent hands from remaining cards.
    
    Args:
        known_cards: Our visible cards
        num_samples: Number of opponent hands to sample
        deck: Full deck
        opponent_hand_size: Usually 2
    
    Returns:
        list of sampled opponent hands
    """
    available = [c for c in deck if c not in known_cards]
    
    if len(available) < opponent_hand_size:
        raise ValueError(f"Not enough available cards: {len(available)} < {opponent_hand_size}")
    
    hands = []
    for _ in range(num_samples):
        hand = tuple(random.sample(available, opponent_hand_size))
        hands.append(hand)
    
    return hands


def compute_card_combo_probability(
    hero_hole: Tuple[str, str],
    opponent_hole: Tuple[str, str],
    board: list[str],
) -> float:
    """
    Probability of specific card combo given observed board.
    
    P(hero_hole, opponent_hole | board) = 1 / C(49, 2)
    
    where 49 = 52 - 2 (board) - 2 (hero), and we're computing
    opponent's hole card probability.
    
    Args:
        hero_hole: Hero's 2 hole cards
        opponent_hole: Opponent's 2 hole cards
        board: Community cards
    
    Returns:
        Probability in [0, 1]
    """
    # Check for card collision
    all_cards = list(hero_hole) + list(opponent_hole) + board
    if len(all_cards) != len(set(all_cards)):
        return 0.0  # Impossible combo (duplicate card)
    
    # Number of remaining cards after board and hero hole
    remaining = 52 - len(board) - len(hero_hole)
    
    if remaining <= 1:
        return 0.0  # No cards left for opponent
    
    # Number of ways to choose 2 cards from remaining
    from math import comb
    num_combos = comb(remaining, 2)  # C(49, 2) typically
    
    # Uniform probability over all possible combos
    return 1.0 / num_combos if num_combos > 0 else 0.0


def weighted_card_sampling(
    hero_hole: Tuple[str, str],
    board: list[str],
    num_samples: int,
    weighting: str = 'uniform',
) -> list[Tuple[str, str]]:
    """
    Sample opponent hands with optional weighting.
    
    Weighting schemes:
        - 'uniform': Equal probability to all hands (default)
        - 'hand_strength': Prefer hands of similar strength to hero
        - 'balanced': Mix of strong and weak opponent hands
    
    Args:
        hero_hole: Hero's hole cards
        board: Community board
        num_samples: Number of opponent hands to sample
        weighting: Weighting scheme
    
    Returns:
        list of sampled opponent hands
    """
    known_cards = set(hero_hole) | set(board)
    available = [c for c in DECK if c not in known_cards]
    
    if weighting == 'uniform':
        # Simple uniform sampling
        hands = []
        for _ in range(num_samples):
            hand = tuple(random.sample(available, 2))
            hands.append(hand)
        return hands
    
    elif weighting == 'hand_strength':
        # TODO: Filter to hands of similar strength (equity within 20%)
        # This would require equity calculation vs hero hand
        return sample_opponent_hands(known_cards, num_samples)
    
    else:
        return sample_opponent_hands(known_cards, num_samples)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Deck State Testing ===")
    deck = DeckState.create_fresh_deck()
    print(f"Initial deck size: {deck.get_remaining_count()}")
    
    hero_hole = deck.deal_hole_cards('hero', 2)
    print(f"Hero dealt: {hero_hole}")
    print(f"Remaining: {deck.get_remaining_count()}")
    
    opponent_samples = sample_opponent_hands(
        known_cards=set(hero_hole),
        num_samples=5,
    )
    print(f"Sample opponent hands: {opponent_samples}")
    
    print("\n=== Enumerate Hands ===")
    board = ['Qs', 'Tc', '9d']
    all_opponent_hands = enumerate_opponent_hands(set(hero_hole) | set(board))
    print(f"Possible opponent hands after Qs Tc 9d: {len(all_opponent_hands)}")
    print(f"First 5: {all_opponent_hands[:5]}")
    
    print("\n=== Card Combo Probability ===")
    prob = compute_card_combo_probability(hero_hole, opponent_samples[0], board)
    print(f"P({hero_hole}, {opponent_samples[0]} | {board}): {prob:.6f}")
