"""
Sequential Action History Modeling for Deep CFR (Phase 3.3).

[PHASE 3.3] Sequential History Encoding - LSTM & Transformer Variants

MOTIVATION
==========

Current Problem:
  Flat embedding of action history loses temporal structure.
  
  Example: check-call vs call-check have same embedding
  Even though flow matters (check-call shows weakness; call-check shows strength).
  
  Network can't learn action dynamics:
    - "bet-call-bet" (strength-weakness-strength) = value bluff
    - "check-check-check" (all weak) = value checking
    - "bet-fold" = successful bluff
  
Solution: Sequential Encoding
  Process action history as a time series.
  Use LSTM or Transformer to learn temporal patterns.
  
ARCHITECTURE
=============

Input: Action sequence
  (batch_size, max_actions, action_feature_dim)
  
  where action_feature_dim includes:
    - Action type (FOLD, CHECK, CALL, RAISE, etc.) [one-hot]
    - Action amount (bet size, normalized)
    - Street (preflop, flop, turn, river) [one-hot]
    - Position (hero, opponent) [one-hot]

Processing:
  1. LSTM layer(s): Capture temporal dependencies
     Input:  (batch, seq_len, action_dim) 
     Output: (batch, seq_len, hidden_dim)
  
  2. Attention (optional): Weight important actions
     - "bet-fold" is high signal
     - "check-check" is low signal
  
  3. Final pooling: Convert sequence → fixed-size vector
     - Last hidden state (captures entire sequence)
     - Or attention-weighted average

Output: Sequence representation
  (batch_size, action_embedding_dim)
  
  Concatenate with card features → full observation
  Feed to policy/value head

COMPARISON WITH FLAT EMBEDDING
================================

Flat (Current):
  - Actions → one-hot → concatenate → ignore temporal order
  - Network sees final state but not how we got there
  - Loss: Can't learn action patterns
  - Speed: O(1) forward pass
  - Params: O(action_history_length) features per hand

Sequential (LSTM):
  - Actions → LSTM → final hidden state
  - Network learns temporal patterns
  - Gain: Understands "check-call vs call-check"
  - Speed: O(seq_len) forward pass (but still fast)
  - Params: O(hidden_dim²) shared across all hands
  - Better generalization across unseen sequences

TIME COMPLEXITY
================
Flat embedding:  O(1) lookup + concatenation
LSTM:            O(seq_len * hidden_dim²) per forward pass
                 ≈ O(10 * 256²) = O(655k ops) per hand
                 ≈ 1ms on GPU for batch of 1024

EXAMPLE FLOW
=============

Round of Poker:
  Hero: A♠ K♠
  Flop: [Q♠ J♦ 9♠]
  
  Action history:
    1. Opponent raises 2x (preflop)
    2. Hero calls
    3. Hero checks (flop)
    4. Opponent bets 1x
    5. Hero raises 2x
    → Current state: Hero's turn, needs decision
  
  Flat embedding:
    [1,0,0,0,0] (raise)
    +[0,1,0,0,0] (call)
    +[0,0,1,0,0] (check)
    +[0,1,0,0,0] (bet)
    +[1,1,0,0,0] (raise)
    Concatenated → ignore order → network sees aggregate
  
  LSTM encoding:
    Input sequence:
      t=0: [action=raise, bet_size=2.0, street=preflop, player=opp]
      t=1: [action=call,  bet_size=2.0, street=preflop, player=hero]
      t=2: [action=check, bet_size=0.0, street=flop,    player=hero]
      t=3: [action=bet,   bet_size=1.0, street=flop,    player=opp]
      t=4: [action=raise, bet_size=2.0, street=flop,    player=hero]
    
    LSTM processes in order:
      h_0 = LSTM(x_0, h_init)  →  captures "opp aggression"
      h_1 = LSTM(x_1, h_0)     →  sees hero called opp's aggression
      h_2 = LSTM(x_2, h_1)     →  hero checked (weakness after call)
      h_3 = LSTM(x_3, h_2)     →  opp continues betting
      h_4 = LSTM(x_4, h_3)     →  hero raises aggressively!
    
    Final h_4 encodes: "hero showed weakness then strength" = value raise
    Network can use this pattern to adjust strategy

---

References:
  - Brownlee (2017): "Understanding LSTM Networks"
  - Vaswani et al. (2017): "Attention is All You Need" (Transformer)
  - Wang et al. (2019): "Effective Approaches to Attention-based NMT"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# ACTION ENCODING
# ============================================================================

@dataclass
class ActionFeatures:
    """Features for a single action in the sequence."""
    
    action_type: str  # 'FOLD', 'CHECK', 'CALL', 'RAISE', 'BET'
    amount: float     # Normalized bet amount (0.0 for check)
    street: str       # 'PREFLOP', 'FLOP', 'TURN', 'RIVER'
    player: str       # 'HERO', 'OPPONENT'
    
    # Mappings
    ACTION_TO_IDX = {
        'FOLD': 0, 'CHECK': 1, 'CALL': 2, 'RAISE': 3, 'BET': 4
    }
    STREET_TO_IDX = {
        'PREFLOP': 0, 'FLOP': 1, 'TURN': 2, 'RIVER': 3
    }
    PLAYER_TO_IDX = {'HERO': 0, 'OPPONENT': 1}
    
    NUM_ACTION_TYPES = 5
    NUM_STREETS = 4
    NUM_PLAYERS = 2
    
    def to_tensor(self) -> torch.Tensor:
        """
        Convert to fixed-size tensor for LSTM input.
        
        Returns:
            Tensor of shape [feature_dim] where:
            [action_one_hot (5)] + [amount (1)] + [street_one_hot (4)] + [player_one_hot (2)]
            = 12-dimensional feature vector
        """
        # Action: one-hot [5]
        action_one_hot = torch.zeros(self.NUM_ACTION_TYPES, dtype=torch.float32)
        action_one_hot[self.ACTION_TO_IDX[self.action_type]] = 1.0
        
        # Amount: scalar [1]
        amount_tensor = torch.tensor([self.amount], dtype=torch.float32)
        
        # Street: one-hot [4]
        street_one_hot = torch.zeros(self.NUM_STREETS, dtype=torch.float32)
        street_one_hot[self.STREET_TO_IDX[self.street]] = 1.0
        
        # Player: one-hot [2]
        player_one_hot = torch.zeros(self.NUM_PLAYERS, dtype=torch.float32)
        player_one_hot[self.PLAYER_TO_IDX[self.player]] = 1.0
        
        # Concatenate: [5 + 1 + 4 + 2] = [12]
        features = torch.cat([
            action_one_hot,
            amount_tensor,
            street_one_hot,
            player_one_hot,
        ])
        
        return features
    
    @staticmethod
    def feature_dim() -> int:
        """Total feature dimension for action encoding."""
        return (
            ActionFeatures.NUM_ACTION_TYPES +
            1 +  # amount
            ActionFeatures.NUM_STREETS +
            ActionFeatures.NUM_PLAYERS
        )


# ============================================================================
# SEQUENTIAL HISTORY ENCODERS
# ============================================================================

class LSTMHistoryEncoder(nn.Module):
    """
    Encode action history sequence using LSTM.
    
    Processes action sequences and outputs a fixed-size representation.
    
    Architecture:
      Input:  (batch, max_actions, action_features=12)
      LSTM:   (batch, max_actions, hidden_dim=256)
      Output: (batch, hidden_dim) [final hidden state]
    
    Key Advantages:
      - Learns temporal dependencies
      - Shared parameters across sequences
      - Handles variable-length sequences
      - Efficient for online/streaming use
    """
    
    def __init__(
        self,
        action_feature_dim: int = 12,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        """
        Args:
            action_feature_dim: Dimension of action features (12 with ActionFeatures)
            hidden_dim: LSTM hidden state dimension
            num_layers: Number of stacked LSTM layers
            dropout: Dropout rate between layers
            bidirectional: If True, processes sequence in both directions
        """
        super().__init__()
        
        self.action_feature_dim = action_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=action_feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        
        output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_dim = output_dim
        
        logger.info(
            f"LSTMHistoryEncoder: action_dim={action_feature_dim}, "
            f"hidden={hidden_dim}, output={output_dim}, num_layers={num_layers}"
        )
    
    def forward(
        self,
        action_sequences: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode action sequences to fixed-size representations.
        
        Args:
            action_sequences: (batch, max_seq_len, action_feature_dim)
                             Padded with zeros for short sequences
            sequence_lengths: (batch,) Optional actual sequence lengths
                             For proper handling of variable-length seqs
        
        Returns:
            (batch, output_dim) Final LSTM hidden state(s)
        """
        if sequence_lengths is not None:
            # Pack sequence (ignore padding)
            packed = nn.utils.rnn.pack_padded_sequence(
                action_sequences,
                sequence_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            # Process entire padded sequence
            _, (h_n, _) = self.lstm(action_sequences)
        
        # h_n shape: (num_layers * num_directions, batch, hidden_dim)
        # For single layer: (1, batch, hidden_dim)
        # For bidirectional: (2, batch, hidden_dim)
        
        # Take final layer output
        h_final = h_n[-1]  # (batch, hidden_dim) or (batch, 2*hidden_dim)
        
        return h_final


class TransformerHistoryEncoder(nn.Module):
    """
    Encode action history using Transformer (attention-based).
    
    Alternative to LSTM: uses self-attention instead of recurrence.
    
    Advantages over LSTM:
      - Parallel computation (faster training & inference)
      - Better at long-range dependencies
      - More modern / better empirical results
    
    Trade-off:
      - Higher memory usage
      - More complex implementation
    
    Architecture:
      Input:  (batch, max_actions, action_features=12)
      Pos Encoding: Add positional information
      Transformer: Self-attention layers
      Output: (batch, action_features) [averaged+pooled]
    """
    
    def __init__(
        self,
        action_feature_dim: int = 12,
        embedding_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ):
        """
        Args:
            action_feature_dim: Dimension of input action features
            embedding_dim: Dimension of Transformer embeddings
            num_heads: Number of attention heads
            num_layers: Number of Transformer layers
            feedforward_dim: Dimension of feedforward networks
            dropout: Dropout rate
        """
        super().__init__()
        
        self.action_feature_dim = action_feature_dim
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim
        
        # Project input to embedding space
        self.input_projection = nn.Linear(action_feature_dim, embedding_dim)
        
        # Positional encoding (learned)
        self.max_seq_len = 100  # Maximum action history length
        self.positional_encoding = nn.Embedding(self.max_seq_len, embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        logger.info(
            f"TransformerHistoryEncoder: action_dim={action_feature_dim}, "
            f"embedding={embedding_dim}, heads={num_heads}, layers={num_layers}"
        )
    
    def forward(
        self,
        action_sequences: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode action sequences using Transformer.
        
        Args:
            action_sequences: (batch, max_seq_len, action_feature_dim)
            sequence_lengths: (batch,) Optional actual sequence lengths
        
        Returns:
            (batch, embedding_dim) Pooled Transformer output
        """
        batch_size, seq_len, _ = action_sequences.shape
        
        # Project to embedding space
        x = self.input_projection(action_sequences)  # (batch, seq_len, embedding_dim)
        
        # Add positional encoding
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_encoding = self.positional_encoding(positions)
        x = x + pos_encoding
        
        # Create attention mask for padding
        if sequence_lengths is not None:
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
        else:
            mask = None
        
        # Transformer forward
        transformed = self.transformer(x, src_key_padding_mask=~mask if mask is not None else None)
        
        # Pool: use last non-padded token or mean
        if sequence_lengths is not None:
            # Use the last valid position for each sequence
            indices = (sequence_lengths - 1).long()
            output = transformed[torch.arange(batch_size), indices]  # (batch, embedding_dim)
        else:
            # Simple mean pooling
            output = transformed.mean(dim=1)  # (batch, embedding_dim)
        
        return output


# ============================================================================
# COMBINED OBSERVATION WITH SEQUENTIAL HISTORY
# ============================================================================

class SequentialObservationBuilder:
    """
    Build observations with sequential action history encoding.
    
    Combines:
      1. Card features (hole cards, board)
      2. Pot odds and stack sizes
      3. LSTM encoding of action history (NEW)
      → Concatenated into full observation
    
    Replaces the old flat action history embedding with sequential encoder.
    """
    
    def __init__(
        self,
        history_encoder: Optional[nn.Module] = None,
        encoder_type: str = 'lstm',
        base_feature_dim: int = 334,  # Dimension without history encoding
    ):
        """
        Args:
            history_encoder: Pre-built LSTM/Transformer encoder
                            If None, creates default LSTMHistoryEncoder
            encoder_type: 'lstm' or 'transformer'
            base_feature_dim: Feature dimension before history (cards, pot odds, etc.)
        """
        self.base_feature_dim = base_feature_dim
        self.encoder_type = encoder_type
        
        if history_encoder is None:
            if encoder_type == 'lstm':
                history_encoder = LSTMHistoryEncoder(
                    action_feature_dim=ActionFeatures.feature_dim(),
                    hidden_dim=256,
                    num_layers=2,
                    dropout=0.1,
                )
            elif encoder_type == 'transformer':
                history_encoder = TransformerHistoryEncoder(
                    action_feature_dim=ActionFeatures.feature_dim(),
                    embedding_dim=256,
                    num_heads=4,
                    num_layers=2,
                )
            else:
                raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.history_encoder = history_encoder
        self.history_embedding_dim = history_encoder.output_dim
        
        # Total observation dimension
        self.obs_dim = base_feature_dim + self.history_embedding_dim
        
        logger.info(
            f"SequentialObservationBuilder: base_dim={base_feature_dim}, "
            f"history_dim={self.history_embedding_dim}, total_obs_dim={self.obs_dim}"
        )
    
    def build_observation(
        self,
        card_features: torch.Tensor,  # (base_feature_dim,)
        action_history: list[ActionFeatures],
        max_seq_len: int = 100,
    ) -> torch.Tensor:
        """
        Build full observation with sequential history encoding.
        
        Args:
            card_features: Card-based features (hole, board, pot odds)
                         Shape: (base_feature_dim,)
            action_history: List of ActionFeatures objects
            max_seq_len: Maximum sequence length (pad/truncate)
        
        Returns:
            Full observation tensor (obs_dim,)
        """
        # Encode action history
        action_tensors = []
        for action in action_history:
            action_tensors.append(action.to_tensor())
        
        if len(action_tensors) == 0:
            # Empty history: pad with zeros
            action_sequence = torch.zeros(
                (1, max_seq_len, ActionFeatures.feature_dim()),
                dtype=torch.float32,
            )
            seq_len = torch.tensor([0])
        else:
            # Stack and pad
            action_seq = torch.stack(action_tensors)  # (seq_len, feature_dim)
            seq_len_actual = len(action_tensors)
            
            if seq_len_actual < max_seq_len:
                # Pad with zeros
                padding = torch.zeros(
                    (max_seq_len - seq_len_actual, ActionFeatures.feature_dim()),
                    dtype=torch.float32,
                )
                action_seq = torch.cat([action_seq, padding], dim=0)
            else:
                # Truncate
                action_seq = action_seq[:max_seq_len]
                seq_len_actual = max_seq_len
            
            action_sequence = action_seq.unsqueeze(0)  # (1, max_seq_len, feature_dim)
            seq_len = torch.tensor([seq_len_actual])
        
        # Encode via LSTM/Transformer
        history_embedding = self.history_encoder(action_sequence, seq_len)
        history_embedding = history_embedding.squeeze(0)  # Remove batch dim
        
        # Concatenate with card features
        full_obs = torch.cat([card_features, history_embedding])
        
        return full_obs


# ============================================================================
# UTILITIES
# ============================================================================

def create_history_encoder(
    encoder_type: str = 'lstm',
    **kwargs,
) -> nn.Module:
    """
    Factory function to create history encoder.
    
    Args:
        encoder_type: 'lstm' or 'transformer'
        **kwargs: Additional arguments for the encoder
    
    Returns:
        Initialized encoder module
    """
    if encoder_type == 'lstm':
        return LSTMHistoryEncoder(**kwargs)
    elif encoder_type == 'transformer':
        return TransformerHistoryEncoder(**kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
