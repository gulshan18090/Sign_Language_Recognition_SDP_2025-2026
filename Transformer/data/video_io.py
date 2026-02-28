"""
Video I/O utilities.
Reads a video file and returns a tensor of uniformly-sampled frames.
Handles: mp4, avi, mov, mkv via OpenCV.
"""
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch


def read_video_frames(
    video_path: str,
    num_frames: int = 16,
    frame_size: Tuple[int, int] = (224, 224),
    temporal_jitter: bool = False,
    seed: int = None,
) -> torch.Tensor:
    """
    Uniformly sample `num_frames` from a video file.

    Args:
        video_path     : path to video file
        num_frames     : how many frames to return
        frame_size     : (H, W) each frame is resized to
        temporal_jitter: if True, add a small random offset to each sample index
        seed           : optional RNG seed for reproducibility

    Returns:
        frames: (num_frames, 3, H, W)  float32 in [0, 1]
    """
    rng = np.random.default_rng(seed)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = max(total_frames, 1)

    # Uniform indices across the video
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=float)

    if temporal_jitter and total_frames > num_frames:
        stride = total_frames / num_frames
        jitter = rng.uniform(-stride * 0.4, stride * 0.4, size=num_frames)
        indices = np.clip(indices + jitter, 0, total_frames - 1)

    indices = indices.astype(int)

    H, W = frame_size
    frames = []
    prev_frame = np.zeros((H, W, 3), dtype=np.uint8)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            frame = prev_frame.copy()
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
            prev_frame = frame
        frames.append(frame)

    cap.release()

    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, 3)
    tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)          # (T, 3, H, W)
    return tensor


def find_video_file(folder: Path, extensions: List[str]) -> Path:
    """Return first video file found in folder, or raise FileNotFoundError."""
    for ext in extensions:
        matches = sorted(folder.glob(f"*{ext}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No video file with extensions {extensions} found in {folder}"
    )
