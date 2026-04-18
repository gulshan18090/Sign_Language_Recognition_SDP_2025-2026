"""
Augmentation Visualization Tool — VIDEO PLAYER edition.

Plays original (top) and augmented (bottom) side-by-side in a single
OpenCV window. No Qt, no matplotlib, no display server needed beyond X11.

Controls:
    SPACE      pause / resume
    R          re-roll augmentation (new random draw)
    S          save current frame pair as PNG
    Q / ESC    quit

Usage:
    python visualize_augmentation.py --video /path/to/sign.mp4
    python visualize_augmentation.py --video /path/to/sign.mp4 --variants 4
    python visualize_augmentation.py --random
    python visualize_augmentation.py --video /path/to/sign.mp4 --fps 8
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Force OpenCV NOT to use its bundled Qt plugin
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

sys.path.insert(0, str(Path(__file__).parent))
from config.config import get_config
from data.augmentation import VideoAugmentor
from data.video_io import find_video_file, read_video_frames


# ═════════════════════════════════════════════════════════════════════════════
# Palette
# ═════════════════════════════════════════════════════════════════════════════
BG          = (14, 14, 22)          # dark background
COL_ORIG    = (255, 106, 124)       # blue-violet  (BGR)
COL_AUG     = (158, 106, 255)       # pink         (BGR)
COL_DROP    = (60,  60,  255)       # red          (BGR)
COL_TEXT    = (240, 240, 232)       # near-white
COL_MUTED   = (138, 107, 107)       # muted
COL_TAG     = (100, 220, 180)       # teal for tags
FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL  = cv2.FONT_HERSHEY_PLAIN


# ═════════════════════════════════════════════════════════════════════════════
# Instrumented augmentor — records what fired
# ═════════════════════════════════════════════════════════════════════════════
class InstrumentedAugmentor(VideoAugmentor):

    def __call__(self, frames: torch.Tensor, is_train: bool = True):
        self.applied: List[str] = []
        self.dropped: List[int] = []
        self.cutout_box: Optional[Tuple[int,int,int,int]] = None
        return super().__call__(frames, is_train)

    def _spatial_augment(self, frames):
        T, C, H, W = frames.shape

        if self.random_crop:
            scale  = random.uniform(*self.crop_scale)
            crop_h, crop_w = int(H * scale), int(W * scale)
            top  = random.randint(0, H - crop_h)
            left = random.randint(0, W - crop_w)
            frames = frames[:, :, top:top+crop_h, left:left+crop_w]
            frames = F.interpolate(frames, size=self.frame_size, mode="bilinear", align_corners=False)
            if scale < 0.99:
                self.applied.append(f"Crop {scale:.2f}")
        else:
            frames = F.interpolate(frames, size=self.frame_size, mode="bilinear", align_corners=False)

        if random.random() < self.flip_p:
            frames = torch.flip(frames, dims=[3])
            self.applied.append("H-Flip")

        if self.cj and random.random() < self.cj_p:
            b = random.uniform(max(0, 1-self.cj_brightness), 1+self.cj_brightness)
            c = random.uniform(max(0, 1-self.cj_contrast),   1+self.cj_contrast)
            s = random.uniform(max(0, 1-self.cj_saturation), 1+self.cj_saturation)
            h = random.uniform(-self.cj_hue, self.cj_hue)
            aug = []
            for t in range(T):
                f = TF.adjust_brightness(frames[t], b)
                f = TF.adjust_contrast(f, c)
                f = TF.adjust_saturation(f, s)
                f = TF.adjust_hue(f, h)
                aug.append(f)
            frames = torch.stack(aug)
            self.applied.append(f"Jitter B{b:.2f} C{c:.2f} S{s:.2f}")

        if random.random() < self.grayscale_p:
            frames = frames.mean(dim=1, keepdim=True).expand_as(frames)
            self.applied.append("Grayscale")

        if random.random() < self.blur_p:
            sigma = random.uniform(0.1, 1.5)
            aug = [TF.gaussian_blur(frames[t],
                   kernel_size=[self.blur_kernel, self.blur_kernel],
                   sigma=sigma) for t in range(T)]
            frames = torch.stack(aug)
            self.applied.append(f"Blur s={sigma:.1f}")

        return frames.clamp(0, 1)

    def _temporal_augment(self, frames):
        T = frames.shape[0]
        if self.frame_drop_p > 0:
            mask = torch.bernoulli(torch.full((T,), 1 - self.frame_drop_p))
            self.dropped = [i for i in range(T) if mask[i] == 0]
            frames = frames * mask.view(T, 1, 1, 1)
            if self.dropped:
                self.applied.append(f"Drop {self.dropped}")
        if random.random() < self.reverse_p:
            frames = torch.flip(frames, dims=[0])
            self.applied.append("Reversed")
        return frames

    def _cutout(self, frames):
        if random.random() >= self.cutout_p:
            return frames
        _, _, H, W = frames.shape
        ch, cw = int(H * self.cutout_size), int(W * self.cutout_size)
        top  = random.randint(0, H - ch)
        left = random.randint(0, W - cw)
        self.cutout_box = (top, left, ch, cw)
        frames = frames.clone()
        frames[:, :, top:top+ch, left:left+cw] = 0
        self.applied.append(f"Cutout ({top},{left})")
        return frames


# ═════════════════════════════════════════════════════════════════════════════
# Frame conversion helpers
# ═════════════════════════════════════════════════════════════════════════════

def tensor_to_bgr(t: torch.Tensor) -> np.ndarray:
    """(3,H,W) float [0,1] → (H,W,3) uint8 BGR for OpenCV."""
    arr = t.permute(1, 2, 0).cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def denorm_to_bgr(t: torch.Tensor, mean, std) -> np.ndarray:
    """Reverse ImageNet normalisation then convert to BGR uint8."""
    t = t.clone().cpu()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return tensor_to_bgr(t)


# ═════════════════════════════════════════════════════════════════════════════
# Drawing helpers
# ═════════════════════════════════════════════════════════════════════════════
DISP_W, DISP_H = 640, 360   # display size per frame


def make_frame_canvas(
    orig_bgr: np.ndarray,
    aug_bgr:  np.ndarray,
    frame_idx: int,
    total_frames: int,
    applied: List[str],
    dropped: List[int],
    cutout_box: Optional[Tuple],
    paused: bool,
    variant_idx: int,
    total_variants: int,
    sidebar_w: int = 300,
) -> np.ndarray:
    """
    Compose the full display frame:
        [orig | aug | sidebar]   ← top
        [      progress bar    ] ← bottom strip
    """
    # Resize both frames to display size
    orig_disp = cv2.resize(orig_bgr, (DISP_W, DISP_H))
    aug_disp  = cv2.resize(aug_bgr,  (DISP_W, DISP_H))

    # ── Dropped frame overlay ─────────────────────────────────────────────────
    if frame_idx in dropped:
        overlay = aug_disp.copy()
        overlay[:] = (0, 0, 180)
        aug_disp = cv2.addWeighted(aug_disp, 0.25, overlay, 0.75, 0)
        cv2.putText(aug_disp, "DROPPED", (DISP_W//2 - 70, DISP_H//2),
                    FONT, 1.2, (80, 80, 255), 2, cv2.LINE_AA)

    # ── CutOut rectangle ──────────────────────────────────────────────────────
    if cutout_box and frame_idx not in dropped:
        top_r, left_r, ch, cw = cutout_box
        oh, ow = orig_bgr.shape[:2]
        sx = DISP_W / ow;  sy = DISP_H / oh
        x1 = int(left_r * sx);  y1 = int(top_r  * sy)
        x2 = int((left_r+cw)*sx); y2 = int((top_r+ch)*sy)
        cv2.rectangle(aug_disp, (x1,y1), (x2,y2), (0,220,220), 2)

    # ── Row labels ────────────────────────────────────────────────────────────
    _label(orig_disp, "ORIGINAL",   COL_ORIG, top=True)
    _label(aug_disp,  "AUGMENTED",  COL_AUG,  top=True)

    # ── Frame counter ─────────────────────────────────────────────────────────
    _label(orig_disp, f"{frame_idx+1}/{total_frames}", COL_MUTED, top=False)
    _label(aug_disp,  f"{frame_idx+1}/{total_frames}", COL_MUTED, top=False)

    # ── Borders ───────────────────────────────────────────────────────────────
    cv2.rectangle(orig_disp, (0,0), (DISP_W-1, DISP_H-1), COL_ORIG, 2)
    cv2.rectangle(aug_disp,  (0,0), (DISP_W-1, DISP_H-1),
                  COL_DROP if frame_idx in dropped else COL_AUG, 2)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    sidebar = np.full((DISP_H, sidebar_w, 3), BG, dtype=np.uint8)
    _draw_sidebar(sidebar, applied, paused, variant_idx, total_variants)

    # ── Stitch horizontally ───────────────────────────────────────────────────
    row = np.hstack([orig_disp, aug_disp, sidebar])

    # ── Progress bar ──────────────────────────────────────────────────────────
    bar_h    = 18
    bar_row  = np.full((bar_h, row.shape[1], 3), (30, 30, 40), dtype=np.uint8)
    fill_w   = int(row.shape[1] * (frame_idx / max(total_frames - 1, 1)))
    cv2.rectangle(bar_row, (0,0), (fill_w, bar_h),
                  COL_AUG if not paused else COL_MUTED, -1)
    pct = int(100 * frame_idx / max(total_frames - 1, 1))
    cv2.putText(bar_row, f"{pct}%",
                (row.shape[1]//2 - 12, 13), FONT_SMALL, 1.0, COL_TEXT, 1)

    return np.vstack([row, bar_row])


def _label(img, text, color, top=True):
    y = 20 if top else img.shape[0] - 8
    cv2.putText(img, text, (8, y), FONT_SMALL, 1.1, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, y), FONT_SMALL, 1.1, color,   1, cv2.LINE_AA)


def _draw_sidebar(sidebar, applied, paused, variant_idx, total_variants):
    W = sidebar.shape[1]

    # Title
    cv2.putText(sidebar, "SLR AUG VIEWER", (10, 22),
                FONT, 0.45, COL_TEXT, 1, cv2.LINE_AA)

    # Variant indicator
    if total_variants > 1:
        vtext = f"Variant {variant_idx+1} / {total_variants}"
        cv2.putText(sidebar, vtext, (10, 40), FONT_SMALL, 0.95, COL_AUG, 1, cv2.LINE_AA)

    # Divider
    cv2.line(sidebar, (10, 48), (W-10, 48), (50, 50, 70), 1)

    # Applied transforms
    cv2.putText(sidebar, "TRANSFORMS:", (10, 64),
                FONT_SMALL, 0.9, COL_MUTED, 1, cv2.LINE_AA)
    if applied:
        for i, t in enumerate(applied):
            y = 82 + i * 20
            if y > sidebar.shape[0] - 30:
                break
            # Tag background pill
            (tw, th), _ = cv2.getTextSize(t, FONT_SMALL, 0.95, 1)
            cv2.rectangle(sidebar, (8, y-13), (12+tw, y+4),
                          (40, 60, 50), -1)
            cv2.putText(sidebar, t, (10, y),
                        FONT_SMALL, 0.95, COL_TAG, 1, cv2.LINE_AA)
    else:
        cv2.putText(sidebar, "  none", (10, 82),
                    FONT_SMALL, 0.9, COL_MUTED, 1, cv2.LINE_AA)

    # Divider
    div_y = sidebar.shape[0] - 72
    cv2.line(sidebar, (10, div_y), (W-10, div_y), (50, 50, 70), 1)

    # Controls legend
    controls = [
        ("SPACE", "pause/resume"),
        ("R",     "re-roll aug"),
        ("S",     "save frame"),
        ("Q/ESC", "quit"),
    ]
    for i, (key, desc) in enumerate(controls):
        y = div_y + 16 + i * 15
        cv2.putText(sidebar, key,  (10,  y), FONT_SMALL, 0.85, COL_ORIG,  1)
        cv2.putText(sidebar, desc, (70, y), FONT_SMALL, 0.85, COL_MUTED, 1)

    # Pause indicator
    if paused:
        cv2.putText(sidebar, "|| PAUSED", (W//2 - 40, sidebar.shape[0] - 8),
                    FONT, 0.45, (80, 200, 255), 1, cv2.LINE_AA)


# ═════════════════════════════════════════════════════════════════════════════
# Build augmented frame list
# ═════════════════════════════════════════════════════════════════════════════
def build_aug(orig_frames, augmentor, mean, std):
    aug_norm = augmentor(orig_frames.clone(), is_train=True)
    orig_bgr = [tensor_to_bgr(orig_frames[i]) for i in range(len(orig_frames))]
    aug_bgr  = [denorm_to_bgr(aug_norm[i], mean, std) for i in range(len(orig_frames))]
    return orig_bgr, aug_bgr


# ═════════════════════════════════════════════════════════════════════════════
# Main player loop
# ═════════════════════════════════════════════════════════════════════════════
def play(video_path: str, cfg, n_variants: int = 1, fps: int = 8):
    print(f"[Visualizer] Loading: {video_path}")
    orig_frames = read_video_frames(
        video_path=video_path,
        num_frames=cfg.video.num_frames,
        frame_size=tuple(cfg.video.frame_size),
        temporal_jitter=False,
    )
    T = len(orig_frames)

    augmentor = InstrumentedAugmentor(
        frame_size=tuple(cfg.video.frame_size),
        mean=tuple(cfg.video.mean), std=tuple(cfg.video.std),
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

    # Pre-generate all variant augmentations
    variants = []
    for v in range(n_variants):
        ob, ab = build_aug(orig_frames, augmentor, cfg.video.mean, cfg.video.std)
        variants.append({
            "orig": ob, "aug": ab,
            "applied": list(augmentor.applied),
            "dropped": list(augmentor.dropped),
            "cutout":  augmentor.cutout_box,
        })
        print(f"[Visualizer] Variant {v+1}: {augmentor.applied or ['(no transforms)']}")

    win_name = "SLR Augmentation Visualizer  [SPACE=pause  R=reroll  S=save  Q=quit]"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # Set a sensible initial window size
    total_w = DISP_W * 2 + 300
    total_h = DISP_H + 18
    cv2.resizeWindow(win_name, total_w, total_h)

    frame_idx    = 0
    variant_idx  = 0
    paused       = False
    delay_ms     = max(1, int(1000 / fps))
    save_counter = 0

    print("[Visualizer] Window open. Controls: SPACE=pause  R=re-roll  S=save  Q/ESC=quit")

    while True:
        v = variants[variant_idx]
        canvas = make_frame_canvas(
            orig_bgr    = v["orig"][frame_idx],
            aug_bgr     = v["aug"][frame_idx],
            frame_idx   = frame_idx,
            total_frames= T,
            applied     = v["applied"],
            dropped     = v["dropped"],
            cutout_box  = v["cutout"],
            paused      = paused,
            variant_idx = variant_idx,
            total_variants = n_variants,
        )

        cv2.imshow(win_name, canvas)
        key = cv2.waitKey(1 if not paused else 30) & 0xFF

        if key in (ord('q'), 27):          # Q or ESC → quit
            break
        elif key == ord(' '):              # SPACE → pause/resume
            paused = not paused
        elif key == ord('r'):              # R → re-roll augmentation
            ob, ab = build_aug(orig_frames, augmentor, cfg.video.mean, cfg.video.std)
            variants[variant_idx] = {
                "orig": ob, "aug": ab,
                "applied": list(augmentor.applied),
                "dropped": list(augmentor.dropped),
                "cutout":  augmentor.cutout_box,
            }
            print(f"[Visualizer] Re-rolled: {augmentor.applied or ['(none)']}")
            frame_idx = 0
        elif key == ord('s'):             # S → save current frame pair
            out = f"aug_frame_{save_counter:03d}.png"
            cv2.imwrite(out, canvas)
            save_counter += 1
            print(f"[Visualizer] Saved frame → {out}")
        elif key == ord('n') and n_variants > 1:  # N → next variant
            variant_idx = (variant_idx + 1) % n_variants
            frame_idx   = 0

        # Advance frame
        if not paused:
            time.sleep(delay_ms / 1000)
            frame_idx = (frame_idx + 1) % T

        # Check window closed
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("[Visualizer] Closed.")


def export_preview_mp4(
    video_path: str,
    cfg,
    out_path: str,
    n_variants: int = 1,
    fps: int = 8,
    variant_idx: int = 0,
):
    """Headless mode: write the composed side-by-side preview to an MP4 file."""
    print(f"[Visualizer] Loading: {video_path}")
    orig_frames = read_video_frames(
        video_path=video_path,
        num_frames=cfg.video.num_frames,
        frame_size=tuple(cfg.video.frame_size),
        temporal_jitter=False,
    )
    T = len(orig_frames)

    augmentor = InstrumentedAugmentor(
        frame_size=tuple(cfg.video.frame_size),
        mean=tuple(cfg.video.mean), std=tuple(cfg.video.std),
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

    variants = []
    for v in range(n_variants):
        ob, ab = build_aug(orig_frames, augmentor, cfg.video.mean, cfg.video.std)
        variants.append({
            "orig": ob, "aug": ab,
            "applied": list(augmentor.applied),
            "dropped": list(augmentor.dropped),
            "cutout":  augmentor.cutout_box,
        })
        print(f"[Visualizer] Variant {v+1}: {augmentor.applied or ['(no transforms)']}")

    variant_idx = max(0, min(int(variant_idx), len(variants) - 1))
    v = variants[variant_idx]

    # Render first frame to get output size
    canvas0 = make_frame_canvas(
        orig_bgr=v["orig"][0],
        aug_bgr=v["aug"][0],
        frame_idx=0,
        total_frames=T,
        applied=v["applied"],
        dropped=v["dropped"],
        cutout_box=v["cutout"],
        paused=False,
        variant_idx=variant_idx,
        total_variants=len(variants),
    )
    h, w = canvas0.shape[:2]

    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for: {out_path}")

    for frame_idx in range(T):
        canvas = make_frame_canvas(
            orig_bgr=v["orig"][frame_idx],
            aug_bgr=v["aug"][frame_idx],
            frame_idx=frame_idx,
            total_frames=T,
            applied=v["applied"],
            dropped=v["dropped"],
            cutout_box=v["cutout"],
            paused=False,
            variant_idx=variant_idx,
            total_variants=len(variants),
        )
        writer.write(canvas)

    writer.release()
    print(f"[Visualizer] Wrote preview MP4 → {out_path}")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="SLR Augmentation Video Player")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  nargs="+", help="Path to video file (spaces in name ok)")
    src.add_argument("--random", action="store_true", help="Pick random sample from dataset")
    parser.add_argument("--variants", type=int, default=1,
                        help="Pre-generate N augmentation variants (cycle with N key)")
    parser.add_argument("--fps", type=int, default=8,
                        help="Playback speed in frames/sec (default 8)")
    parser.add_argument("--headless", action="store_true",
                        help="No GUI window; export an MP4 preview instead")
    parser.add_argument("--out", type=str, default="aug_preview.mp4",
                        help="Output path for --headless export (default: aug_preview.mp4)")
    parser.add_argument("--variant-idx", type=int, default=0,
                        help="Which variant index to export in --headless mode (0-based)")
    args = parser.parse_args()

    cfg = get_config()

    if args.random:
        import pandas as pd
        df = pd.read_csv(cfg.paths.csv_path, sep=cfg.data.csv_sep,
                         header=None, names=cfg.data.csv_columns
                        ).dropna(subset=[cfg.data.id_column])
        videos_root = Path(cfg.paths.videos_root)
        valid = []
        for idd in df[cfg.data.id_column].astype(str):
            folder = videos_root / idd
            if folder.exists():
                try:
                    valid.append(str(find_video_file(folder, cfg.video.video_extensions)))
                except FileNotFoundError:
                    pass
        if not valid:
            print("[Visualizer] No valid videos found. Check cfg.paths.videos_root.")
            sys.exit(1)
        video_path = random.choice(valid)
        print(f"[Visualizer] Random: {video_path}")
    else:
        video_path = " ".join(args.video)

    if args.headless:
        export_preview_mp4(
            video_path=video_path,
            cfg=cfg,
            out_path=args.out,
            n_variants=args.variants,
            fps=args.fps,
            variant_idx=args.variant_idx,
        )
        return

    if not os.environ.get("DISPLAY"):
        print("[Visualizer] No DISPLAY detected (headless session).")
        print("            Re-run with --headless --out aug_preview.mp4")
        sys.exit(2)

    play(video_path, cfg, n_variants=args.variants, fps=args.fps)


if __name__ == "__main__":
    main()