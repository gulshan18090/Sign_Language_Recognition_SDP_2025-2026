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
                 dropout_p=0.1, max_length=config.max_frames, biDirectional=False, encoder_hidden_size=None):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.max_length = max_length
        self.D = 2 if biDirectional else 1
        # Encoder hidden size (for projection if needed)
        self.encoder_hidden_size = encoder_hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn = nn.Linear(hidden_size*2, max_length)
        # Project encoder outputs to match decoder hidden_size for concatenation
        if encoder_hidden_size is not None and encoder_hidden_size != hidden_size:
            self.encoder_proj = nn.Linear(encoder_hidden_size, hidden_size)
        else:
            self.encoder_proj = None
        self.attn_combine = nn.Linear(hidden_size*2, hidden_size)
        self.dropout = nn.Dropout(dropout_p)

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

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[:, 0, :], hidden[0][0]), 1)),
            dim=1
        )

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
