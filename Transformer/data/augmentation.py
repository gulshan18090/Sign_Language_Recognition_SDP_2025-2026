"""
Video augmentation pipeline for sign language recognition.

Spatial augmentations are applied consistently across all frames in a clip
(same crop, same flip, same color transform per sample).

Temporal augmentations operate on the frame sequence axis.

All transforms operate on (T, 3, H, W) float32 tensors in [0, 1].
"""
import random
from typing import Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class VideoAugmentor:
    """
    Composable augmentation pipeline. Call with is_train=True during training.
    Normalise is always applied (train + val + test).
    """

    def __init__(
        self,
        frame_size: Tuple[int, int] = (224, 224),
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std:  Tuple[float, ...] = (0.229, 0.224, 0.225),
        # Spatial
        random_crop: bool = True,
        random_crop_scale: Tuple[float, float] = (0.7, 1.0),
        horizontal_flip_p: float = 0.5,
        color_jitter: bool = True,
        color_jitter_brightness: float = 0.3,
        color_jitter_contrast: float = 0.3,
        color_jitter_saturation: float = 0.2,
        color_jitter_hue: float = 0.1,
        color_jitter_p: float = 0.8,
        grayscale_p: float = 0.1,
        gaussian_blur_p: float = 0.3,
        gaussian_blur_kernel: int = 5,
        # Temporal
        frame_drop_p: float = 0.1,
        temporal_reverse_p: float = 0.1,
        # Regularisation
        cutout_p: float = 0.3,
        cutout_size: float = 0.15,
    ):
        self.frame_size = frame_size
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std  = torch.tensor(std).view(1, 3, 1, 1)

        self.random_crop = random_crop
        self.crop_scale  = random_crop_scale
        self.flip_p      = horizontal_flip_p
        self.cj          = color_jitter
        self.cj_brightness  = color_jitter_brightness
        self.cj_contrast    = color_jitter_contrast
        self.cj_saturation  = color_jitter_saturation
        self.cj_hue         = color_jitter_hue
        self.cj_p           = color_jitter_p
        self.grayscale_p    = grayscale_p
        self.blur_p         = gaussian_blur_p
        self.blur_kernel    = gaussian_blur_kernel if gaussian_blur_kernel % 2 == 1 else gaussian_blur_kernel + 1
        self.frame_drop_p   = frame_drop_p
        self.reverse_p      = temporal_reverse_p
        self.cutout_p       = cutout_p
        self.cutout_size    = cutout_size

    # ── Public ────────────────────────────────────────────────────────────────

    def __call__(self, frames: torch.Tensor, is_train: bool) -> torch.Tensor:
        """
        frames : (T, 3, H, W)  float32 in [0, 1]
        Returns: (T, 3, H, W)  normalised float32
        """
        if is_train:
            frames = self._spatial_augment(frames)
            frames = self._temporal_augment(frames)
            frames = self._cutout(frames)
        else:
            frames = self._center_crop(frames)

        frames = self._normalise(frames)
        return frames

    # ── Spatial (same transform applied to every frame) ───────────────────────

    def _spatial_augment(self, frames: torch.Tensor) -> torch.Tensor:
        T, C, H, W = frames.shape

        # 1. Random resized crop (same region for all frames)
        if self.random_crop:
            scale = random.uniform(*self.crop_scale)
            crop_h = int(H * scale)
            crop_w = int(W * scale)
            top  = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
            frames = frames[:, :, top:top+crop_h, left:left+crop_w]
            frames = F.interpolate(frames, size=self.frame_size, mode="bilinear", align_corners=False)
        else:
            frames = F.interpolate(frames, size=self.frame_size, mode="bilinear", align_corners=False)

        # 2. Random horizontal flip (same for all frames — preserves temporal consistency)
        if random.random() < self.flip_p:
            frames = torch.flip(frames, dims=[3])

        # 3. Color jitter (same params per clip, applied frame by frame)
        if self.cj and random.random() < self.cj_p:
            brightness = random.uniform(max(0, 1 - self.cj_brightness), 1 + self.cj_brightness)
            contrast   = random.uniform(max(0, 1 - self.cj_contrast),   1 + self.cj_contrast)
            saturation = random.uniform(max(0, 1 - self.cj_saturation), 1 + self.cj_saturation)
            hue        = random.uniform(-self.cj_hue, self.cj_hue)
            augmented = []
            for t in range(T):
                f = TF.adjust_brightness(frames[t], brightness)
                f = TF.adjust_contrast(f, contrast)
                f = TF.adjust_saturation(f, saturation)
                f = TF.adjust_hue(f, hue)
                augmented.append(f)
            frames = torch.stack(augmented, dim=0)

        # 4. Random grayscale (replicate across 3 channels)
        if random.random() < self.grayscale_p:
            gray = frames.mean(dim=1, keepdim=True).expand_as(frames)
            frames = gray

        # 5. Gaussian blur (same kernel, applied per frame)
        if random.random() < self.blur_p:
            sigma = random.uniform(0.1, 1.5)
            augmented = []
            for t in range(T):
                f = TF.gaussian_blur(frames[t], kernel_size=[self.blur_kernel, self.blur_kernel], sigma=sigma)
                augmented.append(f)
            frames = torch.stack(augmented, dim=0)

        return frames.clamp(0.0, 1.0)

    def _center_crop(self, frames: torch.Tensor) -> torch.Tensor:
        """Deterministic center crop for val/test."""
        _, _, H, W = frames.shape
        h, w = self.frame_size
        top  = (H - h) // 2
        left = (W - w) // 2
        frames = frames[:, :, top:top+h, left:left+w]
        return F.interpolate(frames, size=self.frame_size, mode="bilinear", align_corners=False)

    # ── Temporal ─────────────────────────────────────────────────────────────

    def _temporal_augment(self, frames: torch.Tensor) -> torch.Tensor:
        T = frames.shape[0]

        # Frame dropout: zero out random frames (simulate occluded/missing frames)
        if self.frame_drop_p > 0:
            mask = torch.bernoulli(torch.full((T,), 1 - self.frame_drop_p))
            frames = frames * mask.view(T, 1, 1, 1)

        # Temporal reversal: flip frame order
        if random.random() < self.reverse_p:
            frames = torch.flip(frames, dims=[0])

        return frames

    # ── CutOut ───────────────────────────────────────────────────────────────

    def _cutout(self, frames: torch.Tensor) -> torch.Tensor:
        """Zero out a random square patch, same location across all frames."""
        if random.random() >= self.cutout_p:
            return frames
        _, _, H, W = frames.shape
        cut_h = int(H * self.cutout_size)
        cut_w = int(W * self.cutout_size)
        top  = random.randint(0, H - cut_h)
        left = random.randint(0, W - cut_w)
        frames = frames.clone()
        frames[:, :, top:top+cut_h, left:left+cut_w] = 0.0
        return frames

    # ── Normalise ────────────────────────────────────────────────────────────

    def _normalise(self, frames: torch.Tensor) -> torch.Tensor:
        """ImageNet normalisation: (T, 3, H, W) → mean/std subtracted."""
        return (frames - self.mean.to(frames.device)) / self.std.to(frames.device)
