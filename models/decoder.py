import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Only show ERROR logs, hide INFO/DEBUG

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, device,
                 dropout_p=0.1, max_length=config.max_frames, biDirectional=False, encoder_hidden_size=None,
                 attn_dropout_p=0.1, attn_temperature=1.0):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.max_length = max_length
        self.D = 2 if biDirectional else 1
        # Encoder hidden size (for projection if needed)
        self.encoder_hidden_size = encoder_hidden_size
        
        # Attention temperature - higher = softer attention (prevents collapse)
        self.attn_temperature = attn_temperature

        self.embedding = nn.Embedding(output_size, hidden_size)
        
        # Improved attention: use separate key/query projections
        self.attn_query = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_key = nn.Linear(encoder_hidden_size or hidden_size, hidden_size)
        self.attn_energy = nn.Linear(hidden_size, 1)
        
        # Legacy attention for backward compatibility
        self.attn = nn.Linear(hidden_size*2, max_length)
        
        # Project encoder outputs to match decoder hidden_size for concatenation
        if encoder_hidden_size is not None and encoder_hidden_size != hidden_size:
            self.encoder_proj = nn.Linear(encoder_hidden_size, hidden_size)
        else:
            self.encoder_proj = None
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        
        # Attention dropout - regularizes attention weights to prevent collapse
        self.attn_dropout = nn.Dropout(attn_dropout_p)

        self.rnn = nn.LSTM(
            hidden_size, hidden_size,
            batch_first=True,
            bidirectional=biDirectional
        )
        self.out = nn.Linear(hidden_size * (2 if biDirectional else 1), output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(input.shape[0], input.shape[1], -1)
        embedded = self.dropout(embedded)

        # Ensure encoder_outputs match expected length
        seq_len = encoder_outputs.shape[1]
        if seq_len != self.max_length:
            if seq_len > self.max_length:
                # Truncate to max_length
                encoder_outputs = encoder_outputs[:, :self.max_length, :]
            else:
                # Pad to max_length
                pad_len = self.max_length - seq_len
                pad = torch.zeros(
                    (encoder_outputs.shape[0], pad_len, encoder_outputs.shape[2]),
                    dtype=encoder_outputs.dtype,
                    device=encoder_outputs.device
                )
                encoder_outputs = torch.cat([encoder_outputs, pad], dim=1)

        # Improved attention mechanism with temperature scaling
        # Query: combination of embedded input and decoder hidden state
        query = self.attn_query(torch.cat((embedded[:, 0, :], hidden[0][0]), 1))  # (B, hidden_size)
        
        # Key: project encoder outputs
        keys = self.attn_key(encoder_outputs)  # (B, T, hidden_size)
        
        # Energy: query-key dot product with learnable transform
        query_expanded = query.unsqueeze(1).expand(-1, keys.size(1), -1)  # (B, T, hidden_size)
        energy = self.attn_energy(torch.tanh(query_expanded + keys)).squeeze(-1)  # (B, T)
        
        # Apply temperature scaling (higher temp = softer distribution)
        energy = energy / self.attn_temperature
        
        # Softmax to get attention weights
        attn_weights = F.softmax(energy, dim=1)
        
        # Apply attention dropout during training (prevents collapse)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Re-normalize after dropout
        attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # (B, 1, encoder_hidden_size)
        
        # Project encoder output dimension to match decoder hidden_size
        encoder_dim = attn_applied.shape[2]
        if encoder_dim != self.hidden_size:
            if self.encoder_proj is not None:
                # Use learnable projection
                attn_applied = self.encoder_proj(attn_applied)  # (B, 1, hidden_size)
            else:
                # Fallback: pad or truncate (not ideal, but works)
                B = attn_applied.shape[0]
                attn_applied_flat = attn_applied.view(B, encoder_dim)
                if encoder_dim < self.hidden_size:
                    # Pad with zeros
                    pad = torch.zeros(B, self.hidden_size - encoder_dim, 
                                    dtype=attn_applied.dtype, device=attn_applied.device)
                    attn_applied_flat = torch.cat([attn_applied_flat, pad], dim=1)
                else:
                    # Truncate (shouldn't happen, but safety)
                    attn_applied_flat = attn_applied_flat[:, :self.hidden_size]
                attn_applied = attn_applied_flat.unsqueeze(1)  # (B, 1, hidden_size)

        output = torch.cat((embedded[:, 0, :], attn_applied[:, 0, :]), 1)
        output = self.attn_combine(output).unsqueeze(1)

        output = F.relu(output)
        output, hidden = self.rnn(output, hidden)

        output = F.log_softmax(self.out(output[:, 0, :]), dim=1)
        return output, hidden, attn_weights
