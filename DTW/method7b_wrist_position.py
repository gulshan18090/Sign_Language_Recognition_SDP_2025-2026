"""
Method 7b: M7 (bone angles + relative lengths) + raw wrist position (2D per hand).

Motivation:
  M7 uses wrist-centred normalisation, which removes ALL positional information.
  Hand location relative to the body is a primary phonological parameter in sign
  language: a sign near the forehead vs near the chest may be a different word
  even if the hand shape is identical. This method adds back a lightweight
  2D wrist-position cue from MediaPipe's image-normalised coordinates BEFORE
  wrist-centering.

Feature layout per frame (212 per hand × 2 = 424 dims):
  [ang_l(190) | rel_len_l(20) | wrist_xy_l(2) |
   ang_r(190) | rel_len_r(20) | wrist_xy_r(2)]

Wrist position encoding:
  - MediaPipe reports (x, y) in normalised image coordinates [0, 1].
  - We subtract 0.5 to centre at the image mid-point → range [-0.5, 0.5].
  - z (depth) is excluded: camera-distance dependent, unreliably comparable.
  - If a hand is absent its wrist position is set to (0.0, 0.0) (centred = absent).

Swap invariance:
  When left/right are swapped, the wrist_xy block is swapped too:
  [ang_r | rel_r | wrist_r | ang_l | rel_l | wrist_l]

DTW cost: L1 mean (same as M7, consistent with scoring).
Score: 1 - mean_L1 over aligned path, clamped to [0, 1].
"""

from __future__ import annotations

import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Reuse bone/angle helpers from method7.
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from method7_angles_rel_len import (  # noqa: E402
    HAND_BONES,
    _angle_deg_vec,
    _bone_vectors_and_lengths,
    _hand_present,
    _pairwise_bone_angles_deg,
    _split_frame126,
)
from similarity.feature_extractor import HandFeatureExtractor  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HAND_FEAT_DIM = 212   # 190 angles + 20 rel_len + 2 wrist_xy  per hand
FEAT_DIM      = 424   # 212 × 2 hands


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _m7_per_hand(hand21x3: np.ndarray) -> np.ndarray:
    """Returns 210-dim M7 features (190 angles + 20 rel_len) for one hand."""
    if not _hand_present(hand21x3):
        return np.zeros(210, dtype=np.float32)
    bones, lens = _bone_vectors_and_lengths(hand21x3)
    ang = _pairwise_bone_angles_deg(bones) / 180.0
    s = float(np.sum(lens))
    rel = (lens / s).astype(np.float32) if s > 0 else np.zeros(20, dtype=np.float32)
    return np.concatenate([ang, rel]).astype(np.float32)


def _wrist_xy(raw_hand21x3: Optional[np.ndarray]) -> np.ndarray:
    """
    Returns 2-dim wrist position centred at image mid-point.
    raw_hand21x3: (21, 3) landmarks BEFORE wrist-centering, or None if absent.
    """
    if raw_hand21x3 is None:
        return np.zeros(2, dtype=np.float32)
    wrist = raw_hand21x3[0, :2].astype(np.float32)  # (x, y) in [0, 1]
    return wrist - 0.5  # centre → [-0.5, 0.5]


def frame_to_feature(
    norm_vec126: np.ndarray,
    raw_left: Optional[np.ndarray],
    raw_right: Optional[np.ndarray],
) -> np.ndarray:
    """
    Build 424-dim feature from:
      norm_vec126 : (126,) wrist-centred, scale-normalised coords (from extractor)
      raw_left    : (21, 3) raw left-hand landmarks before normalisation, or None
      raw_right   : (21, 3) raw right-hand landmarks before normalisation, or None
    """
    left_norm, right_norm = _split_frame126(norm_vec126)

    m7_l = _m7_per_hand(left_norm)    # (210,)
    m7_r = _m7_per_hand(right_norm)   # (210,)
    wx_l = _wrist_xy(raw_left)        # (2,)
    wx_r = _wrist_xy(raw_right)       # (2,)

    return np.concatenate([m7_l, wx_l, m7_r, wx_r]).astype(np.float32)


# ---------------------------------------------------------------------------
# Video extraction — runs MediaPipe directly to capture raw wrist positions
# ---------------------------------------------------------------------------

def extract_video_features(
    video_path: str | Path,
    similarity_threshold: float = 0.99,
) -> Tuple[np.ndarray, List[int]]:
    """
    Extract 424-dim M7b features from a video file.

    Runs MediaPipe Hands on every frame.  Before normalising landmarks,
    captures the raw wrist (x, y) position.  Returns:
      features  : (N, 424) float32
      kept_idx  : list of original frame indices that were kept
    """
    extractor = HandFeatureExtractor(max_hands=2)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return np.zeros((0, FEAT_DIM), dtype=np.float32), []

    raw_lefts: List[Optional[np.ndarray]] = []
    raw_rights: List[Optional[np.ndarray]] = []
    norm_vecs: List[np.ndarray] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get raw (pre-normalisation) landmarks.
        raw_left, raw_right, _ = extractor.extract_landmarks_from_frame(frame_rgb)

        # Build normalised 126-D coordinate vector (same as standard pipeline).
        norm_vec = extractor.landmarks_to_feature_vector(raw_left, raw_right)

        # Skip frames with no hands.
        if not np.any(norm_vec != 0):
            continue

        raw_lefts.append(raw_left)      # None or (21, 3) before normalization
        raw_rights.append(raw_right)
        norm_vecs.append(norm_vec)

    cap.release()

    if not norm_vecs:
        return np.zeros((0, FEAT_DIM), dtype=np.float32), []

    norm_matrix = np.array(norm_vecs, dtype=np.float32)  # (M, 126)

    # Drop redundant consecutive frames using the same similarity check as M7.
    from compare_three_methods import drop_similar_frames  # noqa: E402
    _, kept_rel = drop_similar_frames(norm_matrix, similarity_threshold)

    kept_norms  = norm_matrix[kept_rel]
    kept_rl     = [raw_lefts[i]  for i in kept_rel]
    kept_rr     = [raw_rights[i] for i in kept_rel]

    feats = np.stack(
        [frame_to_feature(kept_norms[i], kept_rl[i], kept_rr[i])
         for i in range(len(kept_rel))],
        axis=0,
    ).astype(np.float32)

    return feats, list(kept_rel)


# ---------------------------------------------------------------------------
# Swap helpers
# Layout: [m7_l(210) | wrist_l(2) | m7_r(210) | wrist_r(2)]
# ---------------------------------------------------------------------------

def swap_hands_feature_matrix(mat: np.ndarray) -> np.ndarray:
    """Swap left↔right hand blocks across all rows of a feature matrix."""
    # left block:  columns [0:212]
    # right block: columns [212:424]
    return np.concatenate([mat[:, 212:424], mat[:, 0:212]], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# DTW  (identical structure to M7)
# ---------------------------------------------------------------------------

@dataclass
class AlignResult:
    score: float
    path: List[Tuple[int, int, float]]
    swapped_used: bool


def _l1_mean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def dtw_global_align(
    ref_feat: np.ndarray,
    user_feat: np.ndarray,
    ref_idx: Sequence[int],
    user_idx: Sequence[int],
    window_ratio: float,
    swap_mode: str = "global",
) -> AlignResult:
    """
    Global DTW with Sakoe-Chiba band and L1 mean cost.
    Score = mean(1 - L1) over aligned frame pairs, clamped to [0, 1].
    If swap_mode='global', also tries left/right swap and picks best.
    """
    n1, n2 = ref_feat.shape[0], user_feat.shape[0]
    if n1 == 0 or n2 == 0:
        return AlignResult(0.0, [], False)

    window_size = int(window_ratio * max(n1, n2))

    def run(user_mat: np.ndarray) -> Tuple[float, List[Tuple[int, int, float]]]:
        dtw = np.full((n1 + 1, n2 + 1), np.inf, dtype=np.float64)
        dtw[0, 0] = 0.0
        for i in range(1, n1 + 1):
            j_lo = max(1, i - window_size)
            j_hi = min(n2 + 1, i + window_size + 1)
            ri = ref_feat[i - 1]
            for j in range(j_lo, j_hi):
                cost = _l1_mean(ri, user_mat[j - 1])
                dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

        path = []
        i, j = n1, n2
        while i > 0 and j > 0:
            path.append((i - 1, j - 1))
            diag = dtw[i - 1, j - 1]
            up   = dtw[i - 1, j]
            left = dtw[i, j - 1]
            m = min(diag, up, left)
            if m == diag:
                i -= 1
                j -= 1
            elif m == up:
                i -= 1
            else:
                j -= 1
        path.reverse()

        sims: List[float] = []
        out: List[Tuple[int, int, float]] = []
        for pi, pj in path:
            d = _l1_mean(ref_feat[pi], user_mat[pj])
            s = max(0.0, 1.0 - d)
            sims.append(s)
            out.append((int(ref_idx[pi]), int(user_idx[pj]), float(s)))
        return (float(np.mean(sims)) if sims else 0.0), out

    score_a, path_a = run(user_feat)
    if swap_mode == "none":
        return AlignResult(score_a, path_a, False)

    user_sw = swap_hands_feature_matrix(user_feat)
    score_b, path_b = run(user_sw)
    if score_b > score_a:
        return AlignResult(score_b, path_b, True)
    return AlignResult(score_a, path_a, False)
