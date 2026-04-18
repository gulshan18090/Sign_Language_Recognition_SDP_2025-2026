"""
Method 4 feature/DTW variants (research utilities).

This module intentionally does not modify existing Method 1/2/3 code.
It provides richer angle features and swap-invariance options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine

FeatureKind = Literal[
    "angles15",          # 15 joint angles per hand (30 total)
    "bones_angles",      # all pairwise bone angles per hand (190 per hand)
    "bones_angles_len",  # pairwise bone angles + bone lengths per hand (210 per hand)
    "bones_angles_len_d1",  # above + first derivative (time) concatenated
]

SwapMode = Literal["none", "global"]
DTWKind = Literal["global", "subseq"]


# MediaPipe Hands canonical bones (20 edges).
HAND_BONES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),# ring
    (0, 17), (17, 18), (18, 19), (19, 20) # pinky
]


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(1.0 - scipy_cosine(a, b))


def _angle_deg_vec(u: np.ndarray, v: np.ndarray) -> float:
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu <= 0.0 or nv <= 0.0:
        return 0.0
    c = float(np.dot(u, v) / (nu * nv))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _angles15_per_hand_deg(hand21x3: np.ndarray) -> np.ndarray:
    """15 joint angles (deg) per hand; see compare_four_methods.py."""
    if hand21x3.shape != (21, 3):
        return np.zeros((15,), dtype=np.float32)

    W = 0
    TH_CMC, TH_MCP, TH_IP, TH_TIP = 1, 2, 3, 4
    I_MCP, I_PIP, I_DIP, I_TIP = 5, 6, 7, 8
    M_MCP, M_PIP, M_DIP, M_TIP = 9, 10, 11, 12
    R_MCP, R_PIP, R_DIP, R_TIP = 13, 14, 15, 16
    P_MCP, P_PIP, P_DIP, P_TIP = 17, 18, 19, 20

    pts = hand21x3.astype(np.float32, copy=False)

    def angle(a: int, b: int, c: int) -> float:
        return _angle_deg_vec(pts[a] - pts[b], pts[c] - pts[b])

    angles = [
        angle(W, TH_CMC, TH_MCP),
        angle(TH_CMC, TH_MCP, TH_IP),
        angle(TH_MCP, TH_IP, TH_TIP),
        angle(W, I_MCP, I_PIP),
        angle(I_MCP, I_PIP, I_DIP),
        angle(I_PIP, I_DIP, I_TIP),
        angle(W, M_MCP, M_PIP),
        angle(M_MCP, M_PIP, M_DIP),
        angle(M_PIP, M_DIP, M_TIP),
        angle(W, R_MCP, R_PIP),
        angle(R_MCP, R_PIP, R_DIP),
        angle(R_PIP, R_DIP, R_TIP),
        angle(W, P_MCP, P_PIP),
        angle(P_MCP, P_PIP, P_DIP),
        angle(P_PIP, P_DIP, P_TIP),
    ]
    return np.array(angles, dtype=np.float32)


def _split_frame126(vec126: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(left21x3, right21x3) from [left63|right63]."""
    left = vec126[:63].reshape(21, 3).astype(np.float32, copy=False)
    right = vec126[63:].reshape(21, 3).astype(np.float32, copy=False)
    return left, right


def _hand_present(hand21x3: np.ndarray) -> bool:
    return float(np.sum(np.abs(hand21x3))) > 0.0


def _bone_vectors(hand21x3: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      bones: (20,3) vectors
      lens:  (20,) lengths
    """
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
    """All pairwise angles between 20 bone vectors: C(20,2)=190."""
    n = bones20x3.shape[0]
    out = np.zeros((n * (n - 1) // 2,), dtype=np.float32)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            out[k] = _angle_deg_vec(bones20x3[i], bones20x3[j])
            k += 1
    return out


def frame126_to_feature(vec126: np.ndarray, kind: FeatureKind) -> np.ndarray:
    left, right = _split_frame126(vec126)

    if kind == "angles15":
        a_left = _angles15_per_hand_deg(left) if _hand_present(left) else np.zeros((15,), dtype=np.float32)
        a_right = _angles15_per_hand_deg(right) if _hand_present(right) else np.zeros((15,), dtype=np.float32)
        return np.concatenate([a_left, a_right]).astype(np.float32, copy=False) / 180.0

    bones_l, lens_l = _bone_vectors(left)
    bones_r, lens_r = _bone_vectors(right)
    ang_l = _pairwise_bone_angles_deg(bones_l)
    ang_r = _pairwise_bone_angles_deg(bones_r)

    if kind == "bones_angles":
        return np.concatenate([ang_l, ang_r]).astype(np.float32, copy=False) / 180.0

    if kind == "bones_angles_len" or kind == "bones_angles_len_d1":
        feat = np.concatenate([ang_l / 180.0, lens_l, ang_r / 180.0, lens_r]).astype(np.float32, copy=False)
        return feat

    raise ValueError(f"Unknown feature kind: {kind}")


def features_from_coords_matrix(coords: np.ndarray, kind: FeatureKind) -> np.ndarray:
    if coords.size == 0:
        return np.zeros((0, 1), dtype=np.float32)
    base = np.stack([frame126_to_feature(coords[i], kind if kind != "bones_angles_len_d1" else "bones_angles_len") for i in range(coords.shape[0])], axis=0)
    base = base.astype(np.float32, copy=False)
    if kind == "bones_angles_len_d1":
        # First derivative (frame-to-frame). Pad first row with zeros.
        d1 = np.zeros_like(base)
        d1[1:] = base[1:] - base[:-1]
        return np.concatenate([base, d1], axis=1).astype(np.float32, copy=False)
    return base


def swap_hands_feature_vector(vec: np.ndarray, kind: FeatureKind) -> np.ndarray:
    """
    Swap left/right blocks.
    - angles15: 15|15
    - bones_angles: 190|190
    - bones_angles_len(+d1): (190+20)|(190+20) and optionally d1 appended with same shape.
    """
    if kind == "angles15":
        if vec.shape[0] != 30:
            return vec
        return np.concatenate([vec[15:], vec[:15]]).astype(np.float32, copy=False)

    if kind == "bones_angles":
        if vec.shape[0] != 380:
            return vec
        return np.concatenate([vec[190:], vec[:190]]).astype(np.float32, copy=False)

    if kind == "bones_angles_len":
        if vec.shape[0] != 420:
            return vec
        left = vec[:210]
        right = vec[210:]
        return np.concatenate([right, left]).astype(np.float32, copy=False)

    if kind == "bones_angles_len_d1":
        # base(420) + d1(420) = 840
        if vec.shape[0] != 840:
            return vec
        base = vec[:420]
        d1 = vec[420:]
        base_sw = swap_hands_feature_vector(base, "bones_angles_len")
        d1_sw = swap_hands_feature_vector(d1, "bones_angles_len")
        return np.concatenate([base_sw, d1_sw]).astype(np.float32, copy=False)

    return vec


def swap_hands_feature_matrix(mat: np.ndarray, kind: FeatureKind) -> np.ndarray:
    out = np.zeros_like(mat)
    for i in range(mat.shape[0]):
        out[i] = swap_hands_feature_vector(mat[i], kind)
    return out


@dataclass
class AlignResult:
    score: float
    path: List[Tuple[int, int, float]]
    swapped_used: bool


def dtw_global_align(
    ref_feat: np.ndarray,
    user_feat: np.ndarray,
    ref_idx: Sequence[int],
    user_idx: Sequence[int],
    window_ratio: float,
    kind: FeatureKind,
    swap_mode: SwapMode,
) -> AlignResult:
    """Global DTW (end-to-end) with cosine scoring on aligned pairs."""
    from scipy.spatial.distance import euclidean

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
                cost = euclidean(ri, user_mat[j - 1])
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
            s = _cos_sim(ref_feat[pi], user_mat[pj])
            sims.append(s)
            out.append((int(ref_idx[pi]), int(user_idx[pj]), float(s)))
        return (float(np.mean(sims)) if sims else 0.0), out

    score_a, path_a = run(user_feat)
    if swap_mode == "none":
        return AlignResult(score_a, path_a, False)

    user_sw = swap_hands_feature_matrix(user_feat, kind)
    score_b, path_b = run(user_sw)
    if score_b > score_a:
        return AlignResult(score_b, path_b, True)
    return AlignResult(score_a, path_a, False)


# Subsequence DTW could be added later for these variants if needed.
