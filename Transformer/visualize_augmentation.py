"""
Augmentation Visualization Tool for the SLR Video Pipeline.

Shows original vs augmented frames side-by-side in a grid layout.
Each augmentation transform is annotated on the augmented frames.

Usage:
    # Visualize a specific video file
    python visualize_augmentation.py --video drive/Video/Cam2/42/sign.mp4

    # Visualize a random sample from the dataset
    python visualize_augmentation.py --random

    # Run N augmentation variants on the same clip (show diversity)
    python visualize_augmentation.py --video path/to/video.mp4 --variants 4

    # Save to file instead of showing
    python visualize_augmentation.py --video path/to/video.mp4 --save output.png
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# ── Must be set BEFORE importing pyplot to avoid Qt/OpenCV conflict ──────────
os.environ["MPLBACKEND"] = "Agg"          # headless matplotlib, no Qt needed
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # stop OpenCV from touching Qt
import matplotlib
matplotlib.use("Agg")                      # redundant but explicit

import cv2
cv2.setNumThreads(1)                       # avoid OpenCV Qt thread collision
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F

# ── Allow running from project root ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from config.config import get_config
from data.augmentation import VideoAugmentor
from data.video_io import find_video_file, read_video_frames


# ═════════════════════════════════════════════════════════════════════════════
# Instrumented augmentor: records which transforms fired
# ═════════════════════════════════════════════════════════════════════════════

class InstrumentedAugmentor(VideoAugmentor):
    """
    Subclass of VideoAugmentor that records every transform applied
    and which frames were dropped, so we can annotate the visualisation.
    """

    def __call__(self, frames: torch.Tensor, is_train: bool = True):
        self.applied_transforms: List[str] = []
        self.dropped_frames: List[int] = []
        self.cutout_region: Optional[Tuple[int,int,int,int]] = None  # top,left,h,w
        self._is_train = is_train
        return super().__call__(frames, is_train)

    # ── Override each internal method to log ─────────────────────────────────

    def _spatial_augment(self, frames: torch.Tensor) -> torch.Tensor:
        T, C, H, W = frames.shape

        # Crop
        if self.random_crop:
            scale  = random.uniform(*self.crop_scale)
            crop_h = int(H * scale)
            crop_w = int(W * scale)
            top    = random.randint(0, H - crop_h)
            left   = random.randint(0, W - crop_w)
            frames = frames[:, :, top:top+crop_h, left:left+crop_w]
            frames = F.interpolate(frames, size=self.frame_size, mode="bilinear", align_corners=False)
            if scale < 0.99:
                self.applied_transforms.append(f"RandomCrop  scale={scale:.2f}")
        else:
            frames = F.interpolate(frames, size=self.frame_size, mode="bilinear", align_corners=False)

        # Flip
        if random.random() < self.flip_p:
            frames = torch.flip(frames, dims=[3])
            self.applied_transforms.append("HorizontalFlip")

        # Color jitter
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
            self.applied_transforms.append(
                f"ColorJitter  B={brightness:.2f} C={contrast:.2f} S={saturation:.2f} H={hue:+.2f}"
            )

        # Grayscale
        if random.random() < self.grayscale_p:
            gray = frames.mean(dim=1, keepdim=True).expand_as(frames)
            frames = gray
            self.applied_transforms.append("Grayscale")

        # Gaussian blur
        if random.random() < self.blur_p:
            sigma = random.uniform(0.1, 1.5)
            augmented = []
            for t in range(T):
                f = TF.gaussian_blur(frames[t],
                                     kernel_size=[self.blur_kernel, self.blur_kernel],
                                     sigma=sigma)
                augmented.append(f)
            frames = torch.stack(augmented, dim=0)
            self.applied_transforms.append(f"GaussianBlur  sigma={sigma:.2f}")

        return frames.clamp(0.0, 1.0)

    def _temporal_augment(self, frames: torch.Tensor) -> torch.Tensor:
        T = frames.shape[0]

        if self.frame_drop_p > 0:
            mask = torch.bernoulli(torch.full((T,), 1 - self.frame_drop_p))
            self.dropped_frames = [i for i in range(T) if mask[i].item() == 0]
            frames = frames * mask.view(T, 1, 1, 1)
            if self.dropped_frames:
                self.applied_transforms.append(
                    f"FrameDrop  frames={self.dropped_frames}"
                )

        if random.random() < self.reverse_p:
            frames = torch.flip(frames, dims=[0])
            self.applied_transforms.append("TemporalReverse")

        return frames

    def _cutout(self, frames: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.cutout_p:
            return frames
        _, _, H, W = frames.shape
        cut_h = int(H * self.cutout_size)
        cut_w = int(W * self.cutout_size)
        top   = random.randint(0, H - cut_h)
        left  = random.randint(0, W - cut_w)
        self.cutout_region = (top, left, cut_h, cut_w)
        frames = frames.clone()
        frames[:, :, top:top+cut_h, left:left+cut_w] = 0.0
        self.applied_transforms.append(
            f"CutOut  pos=({top},{left}) size={cut_h}x{cut_w}"
        )
        return frames


# ═════════════════════════════════════════════════════════════════════════════
# Tensor → numpy for plotting
# ═════════════════════════════════════════════════════════════════════════════

def denorm(tensor: torch.Tensor, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)) -> np.ndarray:
    """Reverse ImageNet normalisation and return (H,W,3) uint8."""
    t = tensor.clone().cpu()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    t = t.permute(1, 2, 0).numpy()
    t = np.clip(t, 0, 1)
    return (t * 255).astype(np.uint8)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """(3,H,W) float [0,1] → (H,W,3) uint8 — for un-normalised frames."""
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


# ═════════════════════════════════════════════════════════════════════════════
# Core visualisation functions
# ═════════════════════════════════════════════════════════════════════════════

def visualize_single(
    video_path: str,
    cfg,
    save_path: Optional[str] = None,
    num_frames: int = None,
):
    """
    Two-row grid:
      Row 1 — Original frames
      Row 2 — Augmented frames (with per-frame annotations)
    Plus a sidebar listing all applied transforms.
    """
    num_frames = num_frames or cfg.video.num_frames

    # ── Load original frames ──────────────────────────────────────────────────
    orig_frames = read_video_frames(
        video_path=video_path,
        num_frames=num_frames,
        frame_size=tuple(cfg.video.frame_size),
        temporal_jitter=False,
    )  # (T, 3, H, W)  [0,1]

    # ── Augment ──────────────────────────────────────────────────────────────
    augmentor = InstrumentedAugmentor(
        frame_size=tuple(cfg.video.frame_size),
        mean=tuple(cfg.video.mean),
        std=tuple(cfg.video.std),
        random_crop=cfg.augment.random_crop,
        random_crop_scale=tuple(cfg.augment.random_crop_scale),
        horizontal_flip_p=cfg.augment.horizontal_flip_p,
        color_jitter=cfg.augment.color_jitter,
        color_jitter_brightness=cfg.augment.color_jitter_brightness,
        color_jitter_contrast=cfg.augment.color_jitter_contrast,
        color_jitter_saturation=cfg.augment.color_jitter_saturation,
        color_jitter_hue=cfg.augment.color_jitter_hue,
        color_jitter_p=cfg.augment.color_jitter_p,
        grayscale_p=cfg.augment.grayscale_p,
        gaussian_blur_p=cfg.augment.gaussian_blur_p,
        gaussian_blur_kernel=cfg.augment.gaussian_blur_kernel,
        frame_drop_p=cfg.augment.frame_drop_p,
        temporal_reverse_p=cfg.augment.temporal_reverse_p,
        cutout_p=cfg.augment.cutout_p,
        cutout_size=cfg.augment.cutout_size,
    )

    aug_frames_norm = augmentor(orig_frames.clone(), is_train=True)  # (T, 3, H, W) normalised

    # ── Convert to numpy ──────────────────────────────────────────────────────
    orig_np = [to_numpy(orig_frames[i])           for i in range(num_frames)]
    aug_np  = [denorm(aug_frames_norm[i], cfg.video.mean, cfg.video.std)
               for i in range(num_frames)]

    # ── Layout ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(max(14, num_frames * 1.6), 7), facecolor="#0e0e16")

    # Grid: 2 rows of frames + 1 narrow right column for legend
    gs = gridspec.GridSpec(
        2, num_frames + 1,
        figure=fig,
        width_ratios=[1] * num_frames + [2.2],
        hspace=0.08,
        wspace=0.04,
        left=0.02, right=0.98, top=0.88, bottom=0.06,
    )

    AX_FG   = "#0e0e16"
    COL_ORIG = "#7c6aff"
    COL_AUG  = "#ff6a9e"
    COL_DROP = "#ff3b3b"
    COL_TEXT = "#e8e8f0"
    COL_MUT  = "#6b6b8a"
    FONT_MONO = {"family": "monospace"}

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"Augmentation Visualizer  ·  {Path(video_path).name}  ·  {num_frames} frames",
        color=COL_TEXT, fontsize=12, fontweight="bold", y=0.97,
        fontfamily="monospace",
    )

    # ── Row labels ────────────────────────────────────────────────────────────
    fig.text(0.005, 0.73, "ORIGINAL", color=COL_ORIG, fontsize=9,
             fontweight="bold", rotation=90, va="center", fontfamily="monospace")
    fig.text(0.005, 0.27, "AUGMENTED", color=COL_AUG, fontsize=9,
             fontweight="bold", rotation=90, va="center", fontfamily="monospace")

    # ── Frame cells ───────────────────────────────────────────────────────────
    for col, (orig_img, aug_img) in enumerate(zip(orig_np, aug_np)):
        is_dropped = col in augmentor.dropped_frames

        # Original row
        ax_o = fig.add_subplot(gs[0, col])
        ax_o.imshow(orig_img)
        ax_o.set_xticks([]); ax_o.set_yticks([])
        for spine in ax_o.spines.values():
            spine.set_edgecolor(COL_ORIG)
            spine.set_linewidth(1.2)
        ax_o.set_facecolor(AX_FG)
        ax_o.set_title(f"{col+1}", color=COL_MUT, fontsize=7, pad=2, **FONT_MONO)

        # Augmented row
        ax_a = fig.add_subplot(gs[1, col])
        ax_a.imshow(aug_img)
        ax_a.set_xticks([]); ax_a.set_yticks([])
        edge_color = COL_DROP if is_dropped else COL_AUG
        for spine in ax_a.spines.values():
            spine.set_edgecolor(edge_color)
            spine.set_linewidth(1.8 if is_dropped else 1.2)
        ax_a.set_facecolor(AX_FG)

        # Annotations on augmented frames
        if is_dropped:
            ax_a.text(0.5, 0.5, "DROPPED", color=COL_DROP,
                      transform=ax_a.transAxes, ha="center", va="center",
                      fontsize=8, fontweight="bold", **FONT_MONO,
                      bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7))

        if augmentor.cutout_region and not is_dropped:
            top_r, left_r, ch, cw = augmentor.cutout_region
            H_img, W_img = orig_img.shape[:2]
            rect = plt.Rectangle(
                (left_r / W_img, 1 - (top_r + ch) / H_img),
                cw / W_img, ch / H_img,
                transform=ax_a.transAxes,
                linewidth=1.5, edgecolor="#ffdd00", facecolor="none",
                linestyle="--",
            )
            ax_a.add_patch(rect)

    # ── Legend / transforms sidebar ───────────────────────────────────────────
    ax_leg = fig.add_subplot(gs[:, num_frames])
    ax_leg.set_facecolor("#12121a")
    ax_leg.set_xticks([]); ax_leg.set_yticks([])
    for spine in ax_leg.spines.values():
        spine.set_edgecolor("#2a2a40")
        spine.set_linewidth(0.8)

    ax_leg.text(0.5, 0.97, "APPLIED TRANSFORMS", color=COL_AUG,
                transform=ax_leg.transAxes, ha="center", va="top",
                fontsize=8, fontweight="bold", **FONT_MONO)

    if augmentor.applied_transforms:
        n = len(augmentor.applied_transforms)
        step = 0.82 / max(n, 1)
        for i, t in enumerate(augmentor.applied_transforms):
            y = 0.87 - i * step
            # Bullet
            ax_leg.text(0.07, y, "◆", color=COL_AUG,
                        transform=ax_leg.transAxes, va="center", fontsize=6)
            # Wrap long lines
            parts = t.split("  ")
            ax_leg.text(0.15, y, parts[0], color=COL_TEXT,
                        transform=ax_leg.transAxes, va="center",
                        fontsize=7, fontweight="bold", **FONT_MONO)
            if len(parts) > 1:
                ax_leg.text(0.15, y - step * 0.4, "  ".join(parts[1:]),
                            color=COL_MUT, transform=ax_leg.transAxes,
                            va="center", fontsize=6, **FONT_MONO)
    else:
        ax_leg.text(0.5, 0.5, "No transforms\napplied this run",
                    color=COL_MUT, transform=ax_leg.transAxes,
                    ha="center", va="center", fontsize=7, **FONT_MONO)

    # Summary counts
    n_applied = len(augmentor.applied_transforms)
    summary = f"{n_applied} transform{'s' if n_applied != 1 else ''}"
    ax_leg.text(0.5, 0.03, summary, color=COL_MUT,
                transform=ax_leg.transAxes, ha="center", va="bottom",
                fontsize=7, **FONT_MONO)

    _finish(fig, save_path)


def visualize_variants(
    video_path: str,
    cfg,
    n_variants: int = 4,
    save_path: Optional[str] = None,
):
    """
    Show N augmented variants of the same clip stacked vertically.
    Each row = one random augmentation draw.
    Row 0 = original.
    """
    num_frames = cfg.video.num_frames

    orig_frames = read_video_frames(
        video_path=video_path,
        num_frames=num_frames,
        frame_size=tuple(cfg.video.frame_size),
        temporal_jitter=False,
    )

    augmentor = InstrumentedAugmentor(
        frame_size=tuple(cfg.video.frame_size),
        mean=tuple(cfg.video.mean),
        std=tuple(cfg.video.std),
        random_crop=cfg.augment.random_crop,
        random_crop_scale=tuple(cfg.augment.random_crop_scale),
        horizontal_flip_p=cfg.augment.horizontal_flip_p,
        color_jitter=cfg.augment.color_jitter,
        color_jitter_brightness=cfg.augment.color_jitter_brightness,
        color_jitter_contrast=cfg.augment.color_jitter_contrast,
        color_jitter_saturation=cfg.augment.color_jitter_saturation,
        color_jitter_hue=cfg.augment.color_jitter_hue,
        color_jitter_p=cfg.augment.color_jitter_p,
        grayscale_p=cfg.augment.grayscale_p,
        gaussian_blur_p=cfg.augment.gaussian_blur_p,
        gaussian_blur_kernel=cfg.augment.gaussian_blur_kernel,
        frame_drop_p=cfg.augment.frame_drop_p,
        temporal_reverse_p=cfg.augment.temporal_reverse_p,
        cutout_p=cfg.augment.cutout_p,
        cutout_size=cfg.augment.cutout_size,
    )

    # Build all rows: orig + N augmented
    rows = []
    rows.append({
        "frames": [to_numpy(orig_frames[i]) for i in range(num_frames)],
        "label": "ORIGINAL",
        "tags": [],
        "dropped": set(),
    })

    for v in range(n_variants):
        aug_norm = augmentor(orig_frames.clone(), is_train=True)
        rows.append({
            "frames": [denorm(aug_norm[i], cfg.video.mean, cfg.video.std)
                       for i in range(num_frames)],
            "label": f"AUG #{v+1}",
            "tags": list(augmentor.applied_transforms),
            "dropped": set(augmentor.dropped_frames),
        })

    n_rows = len(rows)
    AX_FG    = "#0e0e16"
    COL_ORIG = "#7c6aff"
    COL_AUG  = "#ff6a9e"
    COL_DROP = "#ff3b3b"
    COL_TEXT = "#e8e8f0"
    COL_MUT  = "#6b6b8a"
    FONT_MONO = {"family": "monospace"}

    row_colors = [COL_ORIG] + [COL_AUG] * n_variants

    fig, axes = plt.subplots(
        n_rows, num_frames + 1,
        figsize=(max(14, num_frames * 1.5), n_rows * 2.0),
        facecolor=AX_FG,
        gridspec_kw={"width_ratios": [1] * num_frames + [3], "wspace": 0.04, "hspace": 0.1},
    )
    fig.suptitle(
        f"Augmentation Variants  ·  {Path(video_path).name}  ·  {n_variants} draws",
        color=COL_TEXT, fontsize=11, fontweight="bold", y=0.99,
        fontfamily="monospace",
    )

    for r, row in enumerate(rows):
        rc = row_colors[r]
        for c, img in enumerate(row["frames"]):
            ax = axes[r, c]
            is_dropped = c in row["dropped"]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_edgecolor(COL_DROP if is_dropped else rc)
                sp.set_linewidth(1.5)
            ax.set_facecolor(AX_FG)
            if is_dropped:
                ax.text(0.5, 0.5, "✕", color=COL_DROP,
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=14, fontweight="bold")

        # Right label cell
        ax_lbl = axes[r, num_frames]
        ax_lbl.set_facecolor("#12121a")
        ax_lbl.set_xticks([]); ax_lbl.set_yticks([])
        for sp in ax_lbl.spines.values():
            sp.set_edgecolor("#2a2a40"); sp.set_linewidth(0.8)

        ax_lbl.text(0.5, 0.92, row["label"], color=rc,
                    transform=ax_lbl.transAxes, ha="center", va="top",
                    fontsize=8, fontweight="bold", **FONT_MONO)

        tag_text = " · ".join(t.split("  ")[0] for t in row["tags"]) if row["tags"] else "—"
        # Wrap at 28 chars
        wrapped = []
        for word in tag_text.split(" · "):
            if not wrapped or len(wrapped[-1]) + len(word) + 3 > 28:
                wrapped.append(word)
            else:
                wrapped[-1] += " · " + word
        for wi, line in enumerate(wrapped[:4]):
            ax_lbl.text(0.5, 0.72 - wi * 0.18, line, color=COL_MUT,
                        transform=ax_lbl.transAxes, ha="center", va="top",
                        fontsize=6, **FONT_MONO)

    _finish(fig, save_path)


def _finish(fig, save_path):
    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    if not save_path:
        save_path = "aug_preview.png"
        _auto = True
    else:
        _auto = False
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"[Visualizer] Saved → {Path(save_path).resolve()}")
    if _auto:
        print("[Visualizer] Tip: use --save myfile.png to choose a custom output path.")
    plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SLR Augmentation Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, nargs='+', help="Path to a specific video file (quotes optional)")
    src.add_argument("--random", action="store_true", help="Pick a random sample from the dataset")

    parser.add_argument("--variants", type=int, default=1,
                        help="Number of augmentation variants to show (default=1 → side-by-side orig/aug)")
    parser.add_argument("--save", type=str, default=None,
                        help="Save figure to this path instead of showing")
    parser.add_argument("--frames", type=int, default=None,
                        help="Override number of frames to sample (default: from config)")
    args = parser.parse_args()

    cfg = get_config()

    # ── Resolve video path ────────────────────────────────────────────────────
    if args.random:
        import pandas as pd
        df = pd.read_csv(
            cfg.paths.csv_path, sep=cfg.data.csv_sep,
            header=None, names=cfg.data.csv_columns,
        ).dropna(subset=[cfg.data.id_column])
        videos_root = Path(cfg.paths.videos_root)
        valid = []
        for idd in df[cfg.data.id_column].astype(str):
            folder = videos_root / idd
            if folder.exists():
                try:
                    vp = find_video_file(folder, cfg.video.video_extensions)
                    valid.append(str(vp))
                except FileNotFoundError:
                    pass
        if not valid:
            print("[Visualizer] No valid video found in dataset. Check videos_root in config.")
            sys.exit(1)
        video_path = random.choice(valid)
        print(f"[Visualizer] Random sample: {video_path}")
    else:
        # nargs='+' handles filenames with spaces without needing shell quotes
        video_path = " ".join(args.video)

    # ── Run ───────────────────────────────────────────────────────────────────
    if args.variants == 1:
        visualize_single(
            video_path=video_path,
            cfg=cfg,
            save_path=args.save,
            num_frames=args.frames,
        )
    else:
        visualize_variants(
            video_path=video_path,
            cfg=cfg,
            n_variants=args.variants,
            save_path=args.save,
        )


if __name__ == "__main__":
    main()