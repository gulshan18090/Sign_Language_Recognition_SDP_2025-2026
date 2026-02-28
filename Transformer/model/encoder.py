"""
Temporal Transformer Encoder.

Takes per-frame ViT embeddings (B, T, vit_output_dim) and produces
contextual temporal memory (B, T, d_model) via multi-head self-attention
over the frame sequence.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SLREncoder(nn.Module):
    """
    Temporal encoder on top of ViT frame features.
    Input : (B, T, vit_output_dim)
    Output: (B, T, d_model)  — temporal memory for the decoder
    """

    def __init__(
        self,
        vit_output_dim: int = 384,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(vit_output_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable on small data
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )

    def forward(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        src                  : (B, T, vit_output_dim)
        src_key_padding_mask : (B, T)  True = ignore
        Returns memory       : (B, T, d_model)
        """
        x = self.input_proj(src)
        x = self.pos_enc(x)
        return self.transformer(x, src_key_padding_mask=src_key_padding_mask)