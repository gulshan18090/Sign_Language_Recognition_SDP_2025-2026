"""
Method 8: DTW with L1 cost on angle + relative length + palm normal + temporal derivative.

Builds on Method 7 with two additions:
  1. Palm normal vector per hand (3 dims each) — captures palm orientation,
     which pairwise bone angles cannot encode.
  2. Temporal first derivative (d1) appended — captures motion direction,
     distinguishing signs that are identical in static shape but differ in movement.

Feature layout per frame:
  Static block (426 dims):
    [ang_left(190) | rel_len_left(20) | palm_normal_left(3) |
     ang_right(190) | rel_len_right(20) | palm_normal_right(3)]

  d1 block (426 dims):
    [d1 of static block — frame[t] - frame[t-1], zero-padded at t=0]

  Total: 852 dims

DTW cost: L1 mean (same as Method 7, consistent with scoring).
Score: 1 - mean_L1 over aligned path, clamped to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

# MediaPipe Hands canonical bones (20 edges).
HAND_BONES: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # pinky
]

# Palm plane landmark indices (on hand21x3).
# wrist=0, index_MCP=5, pinky_MCP=17
_WRIST = 0
_INDEX_MCP = 5
_PINKY_MCP = 17

STATIC_DIM    = 426              # 213 per hand × 2
HAND_FEAT_DIM = STATIC_DIM // 2  # 213 — dims per hand in static block
FEAT_DIM      = 852              # static + d1


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


def _palm_normal(hand21x3: np.ndarray) -> np.ndarray:
    """
    Unit normal vector to the palm plane.
    Defined by wrist (0), index MCP (5), pinky MCP (17).
    Returns (3,) zero vector if hand absent or degenerate.
    """
    if not _hand_present(hand21x3):
        return np.zeros(3, dtype=np.float32)
    pts = hand21x3.astype(np.float32, copy=False)
    v1 = pts[_INDEX_MCP] - pts[_WRIST]   # wrist → index MCP
    v2 = pts[_PINKY_MCP] - pts[_WRIST]   # wrist → pinky MCP
    normal = np.cross(v1, v2).astype(np.float32)
    norm = float(np.linalg.norm(normal))
    if norm > 0.0:
        normal = normal / norm
    return normal


def frame126_to_static(vec126: np.ndarray) -> np.ndarray:
    """
    Static features for a single frame: 426 dims.
    Layout: [ang_l(190) | rel_len_l(20) | palm_l(3) | ang_r(190) | rel_len_r(20) | palm_r(3)]
    """
    left, right = _split_frame126(vec126)

    def per_hand(hand: np.ndarray) -> np.ndarray:
        if not _hand_present(hand):
            return np.zeros(HAND_FEAT_DIM, dtype=np.float32)
        bones, lens = _bone_vectors_and_lengths(hand)
        ang = _pairwise_bone_angles_deg(bones) / 180.0
        s = float(np.sum(lens))
        rel = (lens / s).astype(np.float32) if s > 0 else np.zeros(20, dtype=np.float32)
        palm = _palm_normal(hand)
        return np.concatenate([ang, rel, palm]).astype(np.float32)

    return np.concatenate([per_hand(left), per_hand(right)]).astype(np.float32)


def features_from_coords_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Input:  coords (N, 126) — raw wrist-centered MediaPipe coordinates.
    Output: features (N, 852) — static (426) + d1 (426).
    """
    if coords.size == 0:
        return np.zeros((0, FEAT_DIM), dtype=np.float32)

    n = coords.shape[0]
    static = np.stack(
        [frame126_to_static(coords[i]) for i in range(n)], axis=0
    ).astype(np.float32)

    d1 = np.zeros_like(static)
    if n > 1:
        d1[1:] = static[1:] - static[:-1]

    return np.concatenate([static, d1], axis=1)


# ---------------------------------------------------------------------------
# Swap helpers
# Layout of the 852-dim vector:
#   [0:213]   left_static
#   [213:426] right_static
#   [426:639] left_d1
#   [639:852] right_d1
# Swapping hands: swap left↔right in both static and d1 blocks.
# ---------------------------------------------------------------------------

def swap_hands_feature_vector(vec: np.ndarray) -> np.ndarray:
    if vec.shape[0] != FEAT_DIM:
        return vec
    H = HAND_FEAT_DIM
    left_s  = vec[0:H]
    right_s = vec[H:STATIC_DIM]
    left_d  = vec[STATIC_DIM:STATIC_DIM + H]
    right_d = vec[STATIC_DIM + H:FEAT_DIM]
    return np.concatenate([right_s, left_s, right_d, left_d])


def swap_hands_feature_matrix(mat: np.ndarray) -> np.ndarray:
    H = HAND_FEAT_DIM
    return np.concatenate([
        mat[:, H:STATIC_DIM],
        mat[:, 0:H],
        mat[:, STATIC_DIM + H:FEAT_DIM],
        mat[:, STATIC_DIM:STATIC_DIM + H],
    ], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# DTW
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
    If swap_mode='global', also tries left/right hand swap and picks best score.
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

        path: List[Tuple[int, int]] = []
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
