"""
Method 6: DTW alignment using cosine-distance cost (1 - cosine_similarity).

Purpose:
  - Method 3 aligns using Euclidean cost but scores using cosine similarity.
  - This mismatch can produce visually "unrelated" alignments.
  - Method 6 makes the DTW objective consistent with the score by using cosine cost.

Input features:
  - Same as Method 3: MediaPipe hand landmark coordinates per frame (126-D),
    already wrist-centered and scale-normalized per hand.

Notes:
  - This module does NOT change existing Method 1–3 code (research comparability).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Method6Result:
    score: float
    path: List[Tuple[int, int, float]]  # (ref_orig_frame_idx, user_orig_frame_idx, cosine_similarity)


def _normalize_rows(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (unit_rows, norms). Zero rows remain zeros."""
    norms = np.linalg.norm(mat, axis=1)
    unit = mat.copy()
    nz = norms > 0
    unit[nz] = unit[nz] / norms[nz, None]
    return unit, norms


def dtw_align_cosine_cost(
    ref_feat: np.ndarray,
    user_feat: np.ndarray,
    ref_idx: Sequence[int],
    user_idx: Sequence[int],
    *,
    window_ratio: float = 0.3,
) -> Method6Result:
    """
    DTW with Sakoe-Chiba band, using cosine-distance cost:
        cost(i,j) = 1 - cos(ref_i, user_j)

    Returns:
      Method6Result(score=mean cosine similarity on DTW path, path with original frame indices)
    """
    n1, n2 = int(ref_feat.shape[0]), int(user_feat.shape[0])
    if n1 <= 0 or n2 <= 0:
        return Method6Result(score=0.0, path=[])

    # Pre-normalize so cosine similarity is a dot product.
    ref_u, ref_norms = _normalize_rows(ref_feat.astype(np.float32, copy=False))
    user_u, user_norms = _normalize_rows(user_feat.astype(np.float32, copy=False))

    window_size = int(window_ratio * max(n1, n2))
    window_size = max(0, window_size)

    dtw = np.full((n1 + 1, n2 + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0.0

    # DP forward pass (banded).
    for i in range(1, n1 + 1):
        j0 = max(1, i - window_size)
        j1 = min(n2, i + window_size) + 1
        ri = ref_u[i - 1]
        has_ref = ref_norms[i - 1] > 0
        for j in range(j0, j1):
            if has_ref and (user_norms[j - 1] > 0):
                cos_sim = float(np.dot(ri, user_u[j - 1]))
                # Clamp minor FP drift.
                if cos_sim > 1.0:
                    cos_sim = 1.0
                elif cos_sim < -1.0:
                    cos_sim = -1.0
                cost = 1.0 - cos_sim
            else:
                cost = 1.0
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # Backtrack for a single path.
    i, j = n1, n2
    path_ij: List[Tuple[int, int]] = []
    while i > 0 and j > 0:
        path_ij.append((i - 1, j - 1))
        a = dtw[i - 1, j - 1]
        b = dtw[i - 1, j]
        c = dtw[i, j - 1]
        m = a
        step = 0
        if b < m:
            m = b
            step = 1
        if c < m:
            m = c
            step = 2
        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    path_ij.reverse()

    if not path_ij:
        return Method6Result(score=0.0, path=[])

    sims: List[float] = []
    out_path: List[Tuple[int, int, float]] = []
    for ii, jj in path_ij:
        if ref_norms[ii] > 0 and user_norms[jj] > 0:
            s = float(np.dot(ref_u[ii], user_u[jj]))
            if s > 1.0:
                s = 1.0
            elif s < -1.0:
                s = -1.0
        else:
            s = 0.0
        sims.append(s)
        out_path.append((int(ref_idx[ii]), int(user_idx[jj]), s))

    return Method6Result(score=float(np.mean(sims)), path=out_path)

