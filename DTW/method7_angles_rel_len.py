"""
Method 7: DTW with L1 cost on angle + relative length features.

Key idea:
  - Use relative bone angles (shape) and relative lengths (scale-invariant).
  - Score uses mean absolute difference (L1), not cosine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# MediaPipe Hands canonical bones (20 edges).
HAND_BONES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20) # pinky
]


def _angle_deg_vec(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu <= 0.0 or nv <= 0.0:
        return 0.0
    c = float(np.dot(u, v) / (nu * nv))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _hand_present(hand21x3: np.ndarray) -> bool:
    return float(np.sum(np.abs(hand21x3))) > 0.0


def _split_frame126(vec126: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    left = vec126[:63].reshape(21, 3).astype(np.float32, copy=False)
    right = vec126[63:].reshape(21, 3).astype(np.float32, copy=False)
    return left, right


def _bone_vectors_and_lengths(hand21x3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    bones = np.zeros((len(HAND_BONES), 3), dtype=np.float32)
    lens = np.zeros((len(HAND_BONES),), dtype=np.float32)
    if not _hand_present(hand21x3):
        return bones, lens
    pts = hand21x3.astype(np.float32, copy=False)
    for i, (a, b) in enumerate(HAND_BONES):
        v = pts[b] - pts[a]
        bones[i] = v
        lens[i] = float(np.linalg.norm(v))
    return bones, lens


def _pairwise_bone_angles_deg(bones20x3: np.ndarray) -> np.ndarray:
    n = bones20x3.shape[0]
    out = np.zeros((n * (n - 1) // 2,), dtype=np.float32)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            out[k] = _angle_deg_vec(bones20x3[i], bones20x3[j])
            k += 1
    return out


def frame126_to_feature(vec126: np.ndarray) -> np.ndarray:
    """
    Feature = [angles_left(190) | rel_len_left(20) | angles_right(190) | rel_len_right(20)]
    - angles are normalized to [0,1] by /180
    - lengths are normalized per-hand by sum(lengths) to be scale invariant
    """
    left, right = _split_frame126(vec126)

    def per_hand(hand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not _hand_present(hand):
            return np.zeros((190,), dtype=np.float32), np.zeros((20,), dtype=np.float32)
        bones, lens = _bone_vectors_and_lengths(hand)
        ang = _pairwise_bone_angles_deg(bones).astype(np.float32, copy=False) / 180.0
        s = float(np.sum(lens))
        if s > 0:
            rel = (lens / s).astype(np.float32, copy=False)
        else:
            rel = np.zeros((20,), dtype=np.float32)
        return ang, rel

    ang_l, rel_l = per_hand(left)
    ang_r, rel_r = per_hand(right)
    return np.concatenate([ang_l, rel_l, ang_r, rel_r]).astype(np.float32, copy=False)


def features_from_coords_matrix(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.zeros((0, 1), dtype=np.float32)
    feats = np.stack([frame126_to_feature(coords[i]) for i in range(coords.shape[0])], axis=0)
    return feats.astype(np.float32, copy=False)


def swap_hands_feature_vector(vec: np.ndarray) -> np.ndarray:
    # layout: [190A+20L | 190A+20L]
    if vec.shape[0] != 420:
        return vec
    left = vec[:210]
    right = vec[210:]
    return np.concatenate([right, left]).astype(np.float32, copy=False)


def swap_hands_feature_matrix(mat: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        out[i] = swap_hands_feature_vector(mat[i])
    return out


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
    Global DTW with L1 mean cost on feature vectors.
    Score = 1 - mean L1 over aligned pairs, clamped to [0,1].
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
            up = dtw[i - 1, j]
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

