"""
Full pipeline:
    Video frames → ViT frame encoder → Temporal encoder → Decoder → Gloss tokens

    frames (B, T, 3, H, W)
         │
    ViTFrameEncoder     ← ImageNet-pretrained ViT (patch16/224)
         │  (B, T, vit_output_dim)
    SLREncoder          ← temporal reasoning over frame sequence
         │  (B, T, d_model)
    SLRDecoder          ← autoregressive gloss generation
         │
    logits (B, T_tgt, vocab_size)
"""
import torch
import torch.nn as nn

from model.vit_backbone import ViTFrameEncoder
from model.encoder import SLREncoder
from model.decoder import SLRDecoder


class SLRSeq2Seq(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        # ViT
        vit_model_name: str = "vit_small_patch16_224",
        vit_pretrained: bool = True,
        vit_output_dim: int = 384,
        # Temporal encoder
        encoder_d_model: int = 256,
        encoder_nhead: int = 8,
        encoder_num_layers: int = 4,
        encoder_dim_feedforward: int = 1024,
        encoder_dropout: float = 0.1,
        # Decoder
        decoder_d_model: int = 256,
        decoder_nhead: int = 8,
        decoder_num_layers: int = 4,
        decoder_dim_feedforward: int = 1024,
        decoder_dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.frame_encoder = ViTFrameEncoder(
            model_name=vit_model_name,
            pretrained=vit_pretrained,
            output_dim=vit_output_dim,
        )
        self.temporal_encoder = SLREncoder(
            vit_output_dim=vit_output_dim,
            d_model=encoder_d_model,
            nhead=encoder_nhead,
            num_layers=encoder_num_layers,
            dim_feedforward=encoder_dim_feedforward,
            dropout=encoder_dropout,
        )
        self.decoder = SLRDecoder(
            vocab_size=vocab_size,
            d_model=decoder_d_model,
            nhead=decoder_nhead,
            num_layers=decoder_num_layers,
            dim_feedforward=decoder_dim_feedforward,
            dropout=decoder_dropout,
            pad_idx=pad_idx,
        )

    def forward(
        self,
        frames: torch.Tensor,        # (B, T, 3, H, W)
        src_mask: torch.Tensor,       # (B, T)  True = pad
        tgt_in: torch.Tensor,         # (B, T_tgt)
        tgt_pad_mask: torch.Tensor,   # (B, T_tgt)  True = pad
    ) -> torch.Tensor:
        memory = self.encode(frames, src_mask)
        T_tgt  = tgt_in.size(1)
        causal = self.decoder.make_causal_mask(T_tgt, tgt_in.device)
        return self.decoder(
            tgt=tgt_in,
            memory=memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_mask,
        )

    def encode(self, frames: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Encode video frames → temporal memory (for inference reuse)."""
        frame_feats = self.frame_encoder(frames)       # (B, T, vit_output_dim)
        return self.temporal_encoder(frame_feats, src_key_padding_mask=src_mask)

    def freeze_backbone(self):
        self.frame_encoder.freeze()

    def unfreeze_backbone(self):
        self.frame_encoder.unfreeze()

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_parameters_total(self) -> int:
        return sum(p.numel() for p in self.parameters())