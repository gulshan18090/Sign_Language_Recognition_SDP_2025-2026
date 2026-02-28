"""
Transformer Decoder: autoregressive gloss token generation.
"""
import torch
import torch.nn as nn
from model.encoder import PositionalEncoding


class SLRDecoder(nn.Module):
    """
    Autoregressive decoder that cross-attends over encoder memory.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc   = PositionalEncoding(d_model, dropout)
        decoder_layer  = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(
            decoder_layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Weight tying: embedding ↔ output projection (reduces overfitting)
        self.output_proj.weight = self.embedding.weight

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        tgt                     : (B, T_tgt)       – token ids
        memory                  : (B, T_src, d_model)
        tgt_mask                : (T_tgt, T_tgt)   – causal mask
        tgt_key_padding_mask    : (B, T_tgt)        – True = pad
        memory_key_padding_mask : (B, T_src)        – True = pad
        Returns logits          : (B, T_tgt, vocab_size)
        """
        x = self.embedding(tgt)
        x = self.pos_enc(x)
        x = self.transformer(
            tgt=x,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_proj(x)

    @staticmethod
    def make_causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular mask to prevent attending to future tokens."""
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()