"""
Slumbot ACPC Match Controller (Phase 5)

Manage head-to-head matches against Slumbot via ACPC protocol.

Key Features:
    1. **Match Loop**: Coordinate hand-by-hand play
    2. **Hand Tracking**: Record results (blinds, stacks, pot)
    3. **Statistics**: Win rate, confidence interval
    4. **Protocol**: ACPC handshake + game state parsing

Reference:
    - ACPC Client: src/evaluation/acpc_client.py
    - Annual Computer Poker Competition (ACPC): annual poker competition
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Import ACPC client (assume partially implemented)
try:
    from src.evaluation.acpc_client import AcpcClient
    ACPC_AVAILABLE = True
except ImportError:
    ACPC_AVAILABLE = False
    logger.warning("ACPC client not available")


# ============================================================================
# HAND RESULTS
# ============================================================================

@dataclass
class HandResult:
    """
    Result of a single poker hand.
    """
    
    hand_number: int
    """Which hand in the match (0-indexed)"""
    
    my_button: bool
    """Whether we were in button position"""
    
    my_hand: Optional[str] = None
    """Our hole cards (e.g., 'AsKd')"""
    
    opponent_hand: Optional[str] = None
    """Opponent's hole cards (if showdown)"""
    
    stack_my_start: float = 0.0
    """Our stack at hand start (in chips)"""
    
    stack_opponent_start: float = 0.0
    """Opponent stack at hand start"""
    
    stack_my_end: float = 0.0
    """Our stack at hand end"""
    
    stack_opponent_end: float = 0.0
    """Opponent stack at hand end"""
    
    winner: str = ""  # "me", "opponent", "tie"
    """Who won the hand"""
    
    chip_delta: float = 0.0
    """Chips won/lost (positive = won)"""
    
    reason: str = ""
    """Why hand ended (fold, showdown, etc.)"""
    
    duration_seconds: float = 0.0
    """Wall-clock time for hand"""
    
    def net_mbb(self, small_blind: float) -> float:
        """Compute chip delta in small blinds."""
        return self.chip_delta / small_blind if small_blind > 0 else 0.0


# ============================================================================
# MATCH STATISTICS
# ============================================================================

@dataclass
class MatchStatistics:
    """
    Aggregated statistics from a match.
    """
    
    num_hands: int = 0
    """Total hands played"""
    
    hands_won: int = 0
    """Number of hands we won"""
    
    hands_lost: int = 0
    """Number of hands we lost"""
    
    hands_tied: int = 0
    """Number of tied hands"""
    
    total_chip_change: float = 0.0
    """Total chip change (positive = profit)"""
    
    win_rate_mbb: float = 0.0
    """Win rate in small blinds per hand"""
    
    profit_interval_95: Tuple[float, float] = (0.0, 0.0)
    """95% confidence interval on profit"""
    
    elapsed_seconds: float = 0.0
    """Total match duration"""
    
    hands_per_minute: float = 0.0
    """Hands played per minute"""
    
    def update(self, result: HandResult, small_blind: float):
        """Update statistics with a new hand result."""
        self.num_hands += 1
        self.total_chip_change += result.chip_delta
        self.elapsed_seconds += result.duration_seconds
        
        if result.winner == "me":
            self.hands_won += 1
        elif result.winner == "opponent":
            self.hands_lost += 1
        else:
            self.hands_tied += 1
        
        # Update metrics
        if self.num_hands > 0:
            self.win_rate_mbb = (
                self.total_chip_change / small_blind / self.num_hands
                if small_blind > 0 else 0.0
            )
            
            if self.elapsed_seconds > 0:
                self.hands_per_minute = self.num_hands / (self.elapsed_seconds / 60)
        
        # Confidence interval (Clopper-Pearson for small samples)
        self._update_confidence_interval(small_blind)
    
    def _update_confidence_interval(self, small_blind: float):
        """Compute 95% confidence interval on win rate."""
        if self.num_hands == 0:
            self.profit_interval_95 = (0.0, 0.0)
            return
        
        # Uses normal approximation (good for n > 30)
        std_error = np.sqrt(
            self.total_chip_change ** 2 / (self.num_hands * (self.num_hands - 1) + 1)
        ) if self.num_hands > 1 else abs(self.total_chip_change)
        
        margin = 1.96 * std_error  # 95% confidence
        
        self.profit_interval_95 = (
            (self.total_chip_change - margin) / small_blind if small_blind > 0 else 0.0,
            (self.total_chip_change + margin) / small_blind if small_blind > 0 else 0.0,
        )
    
    def __str__(self) -> str:
        """Format statistics for display."""
        return (
            f"Hands: {self.num_hands} | "
            f"Won: {self.hands_won} | Lost: {self.hands_lost} | Tied: {self.hands_tied} | "
            f"Win Rate: {self.win_rate_mbb:+.2f} mbb/h | "
            f"95% CI: ({self.profit_interval_95[0]:+.2f}, {self.profit_interval_95[1]:+.2f}) mbb/h | "
            f"Speed: {self.hands_per_minute:.1f} h/min"
        )


# ============================================================================
# MATCH CONTROLLER
# ============================================================================

class MatchController:
    """
    Orchestrate matches against Slumbot via ACPC protocol.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 9001,
        small_blind: float = 1.0,
        big_blind: float = 2.0,
    ):
        """
        Args:
            host: ACPC server host (where Slumbot runs)
            port: ACPC server port
            small_blind: Small blind amount (in chips)
            big_blind: Big blind amount
        """
        self.host = host
        self.port = port
        self.small_blind = small_blind
        self.big_blind = big_blind
        
        self.client: Optional[AcpcClient] = None
        self.hand_results: List[HandResult] = []
        self.match_stats = MatchStatistics()
        
        logger.info(
            f"MatchController initialized: "
            f"{host}:{port} | "
            f"blinds: {small_blind}/{big_blind}"
        )
    
    def connect(self, timeout_seconds: float = 30.0) -> bool:
        """
        Connect to ACPC server (Slumbot).
        
        Args:
            timeout_seconds: Connection timeout
        
        Returns:
            True if connection successful
        """
        if not ACPC_AVAILABLE:
            logger.error("ACPC client not available")
            return False
        
        try:
            logger.info(f"Connecting to {self.host}:{self.port}...")
            self.client = AcpcClient(self.host, self.port)
            
            # Perform ACPC handshake (assumes AcpcClient.handshake() exists)
            if hasattr(self.client, 'handshake'):
                self.client.handshake()
            
            logger.info("Connected and handshake complete")
            return True
        
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close connection to ACPC server."""
        if self.client:
            try:
                self.client.close()
            except Exception as e:
                logger.error(f"Disconnect error: {e}")
            finally:
                self.client = None
    
    def play_match(
        self,
        num_hands: int = 100,
        decision_engine,  # Function(game_state) -> action_index
    ) -> MatchStatistics:
        """
        Play a complete match (multiple hands).
        
        Args:
            num_hands: Number of hands to play
            decision_engine: Function that takes game state and returns action
        
        Returns:
            MatchStatistics with results
        """
        logger.info(f"Starting match: {num_hands} hands")
        self.hand_results.clear()
        self.match_stats = MatchStatistics()
        
        match_start = time.time()
        
        for hand_num in range(num_hands):
            try:
                result = self._play_single_hand(hand_num, decision_engine)
                self.hand_results.append(result)
                self.match_stats.update(result, self.small_blind)
                
                # Log progress
                if (hand_num + 1) % max(1, num_hands // 10) == 0:
                    logger.info(
                        f"Hand {hand_num + 1}/{num_hands}: {self.match_stats}"
                    )
            
            except Exception as e:
                logger.error(f"Hand {hand_num} failed: {e}")
                # Continue playing despite single hand errors
                continue
        
        self.match_stats.elapsed_seconds = time.time() - match_start
        
        logger.info(f"Match complete: {self.match_stats}")
        return self.match_stats
    
    def _play_single_hand(
        self,
        hand_number: int,
        decision_engine,
    ) -> HandResult:
        """
        Play a single hand against Slumbot.
        
        Args:
            hand_number: Hand number (0-indexed)
            decision_engine: Function that makes decisions
        
        Returns:
            HandResult with outcome
        """
        hand_start = time.time()
        result = HandResult(hand_number=hand_number)
        
        if not self.client:
            raise RuntimeError("Not connected to ACPC server")
        
        # Determine position (alternate each hand)
        result.my_button = (hand_number % 2 == 0)
        
        # Get initial match state from server
        match_state = self.client.get_initial_state(hand_number)
        
        # Play hand
        while not match_state.is_terminal():
            # Whose turn?
            player_id = match_state.current_player()
            
            if player_id == 0:  # Our turn
                action = decision_engine(match_state)
                self.client.send_action(action)
            else:  # Opponent's turn
                match_state = self.client.receive_state()
        
        # Parse final result
        result = self._parse_hand_result(match_state, result)
        result.duration_seconds = time.time() - hand_start
        
        return result
    
    def _parse_hand_result(
        self,
        match_state,
        result: HandResult,
    ) -> HandResult:
        """
        Extract hand result from final match state.
        
        Args:
            match_state: Final game state
            result: HandResult to populate
        
        Returns:
            Completed HandResult
        """
        # Extract stacks (depends on ACPC protocol implementation)
        if hasattr(match_state, 'stacks'):
            result.stack_my_end = match_state.stacks[0]
            result.stack_opponent_end = match_state.stacks[1]
        
        # Determine winner
        if hasattr(match_state, 'payoffs'):
            payoff = match_state.payoffs[0]  # Our payoff
            result.chip_delta = payoff
            
            if payoff > 0:
                result.winner = "me"
            elif payoff < 0:
                result.winner = "opponent"
            else:
                result.winner = "tie"
        
        # Try to extract hole cards if showdown
        if hasattr(match_state, 'hole_cards'):
            result.my_hand = match_state.hole_cards[0]
            result.opponent_hand = match_state.hole_cards[1]
        
        # Determine reason (fold, showdown, etc.)
        if hasattr(match_state, 'reason'):
            result.reason = match_state.reason
        elif result.opponent_hand is None:
            result.reason = "opponent_fold"
        else:
            result.reason = "showdown"
        
        return result
    
    def save_results(self, filepath: str):
        """
        Save match results to JSON file.
        
        Args:
            filepath: Path to save results
        """
        data = {
            'statistics': {
                'num_hands': self.match_stats.num_hands,
                'hands_won': self.match_stats.hands_won,
                'hands_lost': self.match_stats.hands_lost,
                'hands_tied': self.match_stats.hands_tied,
                'total_chip_change': self.match_stats.total_chip_change,
                'win_rate_mbb': self.match_stats.win_rate_mbb,
                'profit_interval_95_lower': self.match_stats.profit_interval_95[0],
                'profit_interval_95_upper': self.match_stats.profit_interval_95[1],
                'elapsed_seconds': self.match_stats.elapsed_seconds,
                'hands_per_minute': self.match_stats.hands_per_minute,
            },
            'hands': [
                {
                    'hand_number': r.hand_number,
                    'my_button': r.my_button,
                    'my_hand': r.my_hand,
                    'opponent_hand': r.opponent_hand,
                    'chip_delta': r.chip_delta,
                    'winner': r.winner,
                    'reason': r.reason,
                    'duration_seconds': r.duration_seconds,
                }
                for r in self.hand_results
            ],
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def get_win_rate_confidence(self) -> Tuple[float, float, float]:
        """
        Get win rate + 95% confidence interval.
        
        Returns:
            (win_rate_mbb, lower_bound, upper_bound)
        """
        return (
            self.match_stats.win_rate_mbb,
            self.match_stats.profit_interval_95[0],
            self.match_stats.profit_interval_95[1],
        )


# ============================================================================
# SLUMBOT SPECIFIC ADAPTER
# ============================================================================

class SlumbotMatchAdapter:
    """
    Adapter for Slumbot-specific ACPC parameters.
    """
    
    # Standard ACPC Leduc Hold'em parameters
    LEDUC_SMALL_BLIND = 1
    LEDUC_BIG_BLIND = 2
    LEDUC_STACK = 200
    
    # Slumbot connection
    SLUMBOT_HOST = "poker.cs.ualberta.ca"
    SLUMBOT_PORT = 9000
    
    @staticmethod
    def create_match_vs_slumbot(decision_engine) -> MatchController:
        """
        Create a match controller configured for Slumbot.
        
        Args:
            decision_engine: Function that makes decisions
        
        Returns:
            MatchController ready to use
        """
        controller = MatchController(
            host=SlumbotMatchAdapter.SLUMBOT_HOST,
            port=SlumbotMatchAdapter.SLUMBOT_PORT,
            small_blind=SlumbotMatchAdapter.LEDUC_SMALL_BLIND,
            big_blind=SlumbotMatchAdapter.LEDUC_BIG_BLIND,
        )
        
        return controller
    
    @staticmethod
    def create_local_test_match(decision_engine) -> MatchController:
        """
        Create a local test match (for development).
        
        Args:
            decision_engine: Function that makes decisions
        
        Returns:
            MatchController for localhost
        """
        controller = MatchController(
            host="localhost",
            port=9001,
            small_blind=1.0,
            big_blind=2.0,
        )
        
        return controller


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Slumbot ACPC Match Controller ===")
    
    # Create mock decision engine for testing
    def mock_decision_engine(game_state):
        """Mock AI that always calls."""
        return 1  # Action index for "call"
    
    # Create local test match
    controller = SlumbotMatchAdapter.create_local_test_match(mock_decision_engine)
    
    print(f"Match controller ready (offline mode): {controller}")
    print(f"Connect with: controller.connect()")
    print(f"Play with: controller.play_match(num_hands=10, decision_engine=fn)")
    
    # Example statistics (for testing without connection)
    stats = MatchStatistics(
        num_hands=50,
        hands_won=28,
        hands_lost=20,
        hands_tied=2,
        total_chip_change=56.0,
    )
    stats.update(
        HandResult(hand_number=0, chip_delta=1.0, winner="me"),
        small_blind=1.0
    )
    
    print(f"\nExample statistics: {stats}")
