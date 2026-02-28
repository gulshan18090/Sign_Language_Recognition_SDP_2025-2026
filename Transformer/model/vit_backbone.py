"""
ViT Frame Encoder.

Takes a batch of video frames (B*T, 3, H, W) and returns per-frame embeddings
(B*T, output_dim) using a pretrained ViT from timm.

The [CLS] token representation is used as the frame-level feature.
"""
import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError("timm is required: pip install timm")


class ViTFrameEncoder(nn.Module):
    """
    Wraps a timm ViT model to encode individual frames.
    Processes all T frames of a clip in parallel as a single large batch.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        output_dim: int = 384,
    ):
        super().__init__()
        # Load backbone (no classification head — num_classes=0 returns embed dim)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,      # remove classifier head → returns CLS token
        )
        backbone_dim = self.backbone.num_features

        # Optional projection if backbone_dim != output_dim
        if backbone_dim != output_dim:
            self.proj = nn.Linear(backbone_dim, output_dim)
        else:
            self.proj = nn.Identity()

        self.output_dim = output_dim

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames : (B, T, 3, H, W)
        Returns: (B, T, output_dim)
        """
        B, T, C, H, W = frames.shape
        # Flatten to (B*T, 3, H, W) — process all frames in one forward pass
        x = frames.view(B * T, C, H, W)
        x = self.backbone(x)          # (B*T, backbone_dim)
        x = self.proj(x)              # (B*T, output_dim)
        return x.view(B, T, self.output_dim)

    def freeze(self) -> None:
        """Freeze all ViT parameters (call during warm-up phase)."""
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        print("[ViTFrameEncoder] Backbone frozen.")

    def unfreeze(self) -> None:
        """Unfreeze all ViT parameters."""
        for p in self.backbone.parameters():
            p.requires_grad_(True)
        print("[ViTFrameEncoder] Backbone unfrozen.")