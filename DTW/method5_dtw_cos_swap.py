"""
Method 5: DTW + cosine on MediaPipe hand landmark coordinates with global L/R hand-swap invariance.

Why this exists:
  - Our base coordinate feature is 126-D per frame: [left(63) | right(63)].
  - If the same sign is performed with the opposite hand, one video often has the signal
    in left(63) while the other has it in right(63). A straight cosine/DTW then compares
    non-zero vs zeros and tanks similarity.

Method 5 fixes this (minimally) by trying DTW scoring with:
  - user as-is
  - user with halves swapped: [right | left]
and picking the better score (global decision per pair).

Note:
  - This file intentionally does NOT modify Method 1–3 code paths (research comparability).
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


FEATURES_PER_HAND = 63
TOTAL_FEATURES = 126


def swap_hands_matrix(mat: np.ndarray) -> np.ndarray:
    """Swap [left(63) | right(63)] -> [right | left] for every frame."""
    if mat.ndim != 2 or mat.shape[1] != TOTAL_FEATURES:
        raise ValueError(f"Expected matrix shape (N,{TOTAL_FEATURES}), got {mat.shape}")
    return np.concatenate([mat[:, FEATURES_PER_HAND:], mat[:, :FEATURES_PER_HAND]], axis=1)


def swap_hands_vector(vec: np.ndarray) -> np.ndarray:
    """Swap a single 126-D feature vector."""
    if vec.ndim != 1 or vec.shape[0] != TOTAL_FEATURES:
        raise ValueError(f"Expected vector shape ({TOTAL_FEATURES},), got {vec.shape}")
    return np.concatenate([vec[FEATURES_PER_HAND:], vec[:FEATURES_PER_HAND]], axis=0)


@dataclass(frozen=True)
class Method5Result:
    score: float
    path: List[Tuple[int, int, float]]
    variant: str  # "orig" or "swap"


def dtw_best_of_user_swap(
    ref_feat: np.ndarray,
    user_feat: np.ndarray,
    ref_idx: Sequence[int],
    user_idx: Sequence[int],
    *,
    window_ratio: float = 0.3,
    verbose: bool = False,
) -> Method5Result:
    """
    Run DTW alignment twice (orig vs swapped user) and keep the better.

    Returns:
      Method5Result(score, path, variant)
    """
    # Lazy import so this module stays lightweight when imported elsewhere.
    from compare_three_methods import dtw_align_with_prefiltered_frames

    def _run(a: np.ndarray, b: np.ndarray) -> Tuple[float, List[Tuple[int, int, float]]]:
        if verbose:
            s, p, _ = dtw_align_with_prefiltered_frames(a, b, ref_idx, user_idx, window_ratio=window_ratio)
            return float(s), [(int(x), int(y), float(z)) for (x, y, z) in p]
        # dtw_align_with_prefiltered_frames prints a lot; default to quiet.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            s, p, _ = dtw_align_with_prefiltered_frames(a, b, ref_idx, user_idx, window_ratio=window_ratio)
        return float(s), [(int(x), int(y), float(z)) for (x, y, z) in p]

    score_orig, path_orig = _run(ref_feat, user_feat)
    score_swap, path_swap = _run(ref_feat, swap_hands_matrix(user_feat))

    if score_swap > score_orig:
        return Method5Result(score=score_swap, path=path_swap, variant="swap")
    return Method5Result(score=score_orig, path=path_orig, variant="orig")

