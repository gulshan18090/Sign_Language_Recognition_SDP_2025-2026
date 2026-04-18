"""
Compare four similarity methods (research extension; does not modify compare_three_methods.py):
1) Regular Cosine on coordinate features (flatten)
2) Frame-wise Cosine on coordinate features (index-to-index)
3) DTW Hybrid on coordinate features (global DTW + local cosine optimization)
4) DTW Hybrid on joint-angle (degree) features + hand-swap invariance

Method 4 idea:
- Convert each frame's MediaPipe landmarks into joint angles (in degrees).
- Run DTW on angle vectors (Euclidean cost, Sakoe-Chiba band like Method 3).
- Score aligned pairs by cosine similarity, but take max(sim(user), sim(swap(user))) so
  left-hand vs right-hand performance is treated as equivalent.

Usage:
  venv310\\Scripts\\python.exe compare_four_methods.py --folder "Bu gün hava çox soyuqdur" --user-idx 0
  venv310\\Scripts\\python.exe compare_four_methods.py --limit-folders 3 --user-idx 0
"""

from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import cosine as scipy_cosine

# Fix Unicode output on Windows console
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "Videos")
MATRICES_DIR = os.path.join(BASE_DIR, "matrices")

sys.path.append(BASE_DIR)
from compare_three_methods import (  # noqa: E402
    extract_all_frames_and_filter,
    dtw_align_with_prefiltered_frames,
    method1_regular_cosine,
    method2_frame_wise_cosine,
)


def safe_folder_name(name: str, max_len: int = 60) -> str:
    s = unicodedata.normalize("NFKD", str(name)).strip()
    s = s.replace(" ", "_")
    return s[:max_len]


def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at point b formed by points (a, b, c) in degrees."""
    ba = a - b
    bc = c - b
    nba = float(np.linalg.norm(ba))
    nbc = float(np.linalg.norm(bc))
    if nba <= 0.0 or nbc <= 0.0:
        return 0.0
    cosang = float(np.dot(ba, bc) / (nba * nbc))
    cosang = max(-1.0, min(1.0, cosang))
    return float(np.degrees(np.arccos(cosang)))


def _angles_per_hand_deg(hand21x3: np.ndarray) -> np.ndarray:
    """
    Compute 15 joint angles (degrees) for one hand:
    - Thumb: wrist-cmc-mcp, cmc-mcp-ip, mcp-ip-tip
    - Index/Middle/Ring/Pinky: wrist-mcp-pip, mcp-pip-dip, pip-dip-tip
    """
    if hand21x3.shape != (21, 3):
        return np.zeros((15,), dtype=np.float32)

    W = 0
    # Thumb
    TH_CMC, TH_MCP, TH_IP, TH_TIP = 1, 2, 3, 4
    # Index
    I_MCP, I_PIP, I_DIP, I_TIP = 5, 6, 7, 8
    # Middle
    M_MCP, M_PIP, M_DIP, M_TIP = 9, 10, 11, 12
    # Ring
    R_MCP, R_PIP, R_DIP, R_TIP = 13, 14, 15, 16
    # Pinky
    P_MCP, P_PIP, P_DIP, P_TIP = 17, 18, 19, 20

    pts = hand21x3.astype(np.float32, copy=False)
    angles = [
        _angle_deg(pts[W], pts[TH_CMC], pts[TH_MCP]),
        _angle_deg(pts[TH_CMC], pts[TH_MCP], pts[TH_IP]),
        _angle_deg(pts[TH_MCP], pts[TH_IP], pts[TH_TIP]),
        _angle_deg(pts[W], pts[I_MCP], pts[I_PIP]),
        _angle_deg(pts[I_MCP], pts[I_PIP], pts[I_DIP]),
        _angle_deg(pts[I_PIP], pts[I_DIP], pts[I_TIP]),
        _angle_deg(pts[W], pts[M_MCP], pts[M_PIP]),
        _angle_deg(pts[M_MCP], pts[M_PIP], pts[M_DIP]),
        _angle_deg(pts[M_PIP], pts[M_DIP], pts[M_TIP]),
        _angle_deg(pts[W], pts[R_MCP], pts[R_PIP]),
        _angle_deg(pts[R_MCP], pts[R_PIP], pts[R_DIP]),
        _angle_deg(pts[R_PIP], pts[R_DIP], pts[R_TIP]),
        _angle_deg(pts[W], pts[P_MCP], pts[P_PIP]),
        _angle_deg(pts[P_MCP], pts[P_PIP], pts[P_DIP]),
        _angle_deg(pts[P_PIP], pts[P_DIP], pts[P_TIP]),
    ]
    return np.array(angles, dtype=np.float32)


def _frame126_to_angles30_deg(vec126: np.ndarray) -> np.ndarray:
    """
    Convert a 126-D coordinate vector (left63|right63) into a 30-D angle vector (left15|right15).
    Angles are in degrees, normalized to [0..1] by dividing by 180 to make DTW Euclidean cost stable.
    """
    if vec126.shape[0] != 126:
        return np.zeros((30,), dtype=np.float32)

    left = vec126[:63].reshape(21, 3)
    right = vec126[63:].reshape(21, 3)
    left_present = float(np.sum(np.abs(left))) > 0.0
    right_present = float(np.sum(np.abs(right))) > 0.0
    a_left = _angles_per_hand_deg(left) if left_present else np.zeros((15,), dtype=np.float32)
    a_right = _angles_per_hand_deg(right) if right_present else np.zeros((15,), dtype=np.float32)
    angles = np.concatenate([a_left, a_right]).astype(np.float32, copy=False)
    return angles / 180.0


def angles_matrix_from_coords(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return np.zeros((0, 30), dtype=np.float32)
    out = np.zeros((coords.shape[0], 30), dtype=np.float32)
    for i in range(coords.shape[0]):
        out[i] = _frame126_to_angles30_deg(coords[i])
    return out


def swap_hands_angles(vec30: np.ndarray) -> np.ndarray:
    """Swap left/right 15-angle blocks."""
    if vec30.shape[0] != 30:
        return vec30
    return np.concatenate([vec30[15:], vec30[:15]]).astype(np.float32, copy=False)


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return float(1.0 - scipy_cosine(a, b))


@dataclass
class Method4Result:
    score: float
    path: List[Tuple[int, int, float]]
    swapped: List[bool]


def dtw_align_angles_with_hand_swap(
    trans_angles: np.ndarray,
    user_angles: np.ndarray,
    trans_idx: List[int],
    user_idx: List[int],
    window_ratio: float = 0.3,
) -> Method4Result:
    """
    Global DTW alignment on angle vectors, with hand-swap invariance:
      cost(i,j) = min(||t_i - u_j||, ||t_i - swap(u_j)||)
      sim(i,j)  = max(cos(t_i, u_j), cos(t_i, swap(u_j)))
    """
    from scipy.spatial.distance import euclidean

    n1 = trans_angles.shape[0]
    n2 = user_angles.shape[0]
    if n1 == 0 or n2 == 0:
        return Method4Result(0.0, [], [])

    window_size = int(window_ratio * max(n1, n2))
    dtw = np.full((n1 + 1, n2 + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0

    user_swapped = np.zeros_like(user_angles)
    for j in range(n2):
        user_swapped[j] = swap_hands_angles(user_angles[j])

    for i in range(1, n1 + 1):
        j_lo = max(1, i - window_size)
        j_hi = min(n2 + 1, i + window_size + 1)
        ti = trans_angles[i - 1]
        for j in range(j_lo, j_hi):
            uj = user_angles[j - 1]
            us = user_swapped[j - 1]
            cost = min(euclidean(ti, uj), euclidean(ti, us))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    # backtrack
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
    out_path: List[Tuple[int, int, float]] = []
    swapped_flags: List[bool] = []
    for i1, j1 in path:
        ti = trans_angles[i1]
        uj = user_angles[j1]
        us = user_swapped[j1]
        sim_u = cos_sim(ti, uj)
        sim_s = cos_sim(ti, us)
        if sim_s > sim_u:
            sim = sim_s
            swapped = True
        else:
            sim = sim_u
            swapped = False
        sims.append(sim)
        out_path.append((int(trans_idx[i1]), int(user_idx[j1]), float(sim)))
        swapped_flags.append(swapped)

    score = float(np.mean(sims)) if sims else 0.0
    return Method4Result(score=score, path=out_path, swapped=swapped_flags)


def run_one_folder(folder_name: str, user_idx: int, window_ratio: float) -> Dict:
    trans_videos = sorted(glob.glob(os.path.join(VIDEOS_DIR, folder_name, "translator_*.mp4")))
    user_videos = sorted(glob.glob(os.path.join(VIDEOS_DIR, folder_name, "user_*.mp4")))
    if not trans_videos or not user_videos:
        raise FileNotFoundError(f"Missing translator/user videos in {folder_name}")
    if user_idx < 0 or user_idx >= len(user_videos):
        raise IndexError(f"user-idx out of range: {user_idx} (0..{len(user_videos)-1})")

    trans_path = trans_videos[0]
    user_path = user_videos[user_idx]

    print("=" * 80)
    print(f"FOLDER: {folder_name}")
    print(f"Translator: {os.path.basename(trans_path)}")
    print(f"User:       {os.path.basename(user_path)}")
    print("-" * 80)

    # Extract coordinate features once (same as compare_three_methods)
    trans_coords, trans_idx = extract_all_frames_and_filter(trans_path, similarity_threshold=0.99)
    user_coords, user_idx_list = extract_all_frames_and_filter(user_path, similarity_threshold=0.99)
    if len(trans_coords) == 0 or len(user_coords) == 0:
        return {"folder": folder_name, "error": "no features"}

    # Methods 1-3 (existing, coordinate features)
    m1 = float(method1_regular_cosine(trans_coords, user_coords))
    m2 = float(method2_frame_wise_cosine(trans_coords, user_coords))
    m3_score, m3_path, _m3_sims = dtw_align_with_prefiltered_frames(
        trans_coords, user_coords, trans_idx, user_idx_list, window_ratio=window_ratio
    )

    # Method 4 (new, angle features + swap invariance)
    trans_ang = angles_matrix_from_coords(trans_coords)
    user_ang = angles_matrix_from_coords(user_coords)
    m4 = dtw_align_angles_with_hand_swap(trans_ang, user_ang, trans_idx, user_idx_list, window_ratio=window_ratio)

    print(f"Method 1 (Flatten Cosine coords): {m1:.4f}")
    print(f"Method 2 (Framewise Cosine coords): {m2:.4f}")
    print(f"Method 3 (DTW+Cos coords): {float(m3_score):.4f}  aligned_pairs={len(m3_path)}")
    print(f"Method 4 (DTW+Cos angles + swap): {float(m4.score):.4f}  aligned_pairs={len(m4.path)}  swapped_pairs={sum(1 for s in m4.swapped if s)}")

    # Save details
    safe = safe_folder_name(folder_name)
    out_dir = os.path.join(MATRICES_DIR, safe)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "method4_angles_swap.json")
    payload = {
        "folder": folder_name,
        "translator": os.path.basename(trans_path),
        "user": os.path.basename(user_path),
        "params": {"window_ratio": window_ratio, "similarity_threshold": 0.99},
        "method1": {"score": m1},
        "method2": {"score": m2},
        "method3": {"score": float(m3_score), "aligned_pairs": len(m3_path)},
        "method4": {
            "score": float(m4.score),
            "aligned_pairs": len(m4.path),
            "swapped_pairs": int(sum(1 for s in m4.swapped if s)),
            "path": m4.path[:200],  # keep file size reasonable
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved: {out_path}")

    return payload


def discover_folders(limit_folders: Optional[int]) -> List[str]:
    folders = sorted(
        d for d in os.listdir(VIDEOS_DIR)
        if os.path.isdir(os.path.join(VIDEOS_DIR, d))
        and glob.glob(os.path.join(VIDEOS_DIR, d, "translator_*.mp4"))
        and glob.glob(os.path.join(VIDEOS_DIR, d, "user_*.mp4"))
    )
    if limit_folders is not None:
        folders = folders[:limit_folders]
    return folders


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare 4 methods (adds angle+swap DTW as Method 4).")
    p.add_argument("--folder", type=str, default=None, help="Folder under Videos/ to run.")
    p.add_argument("--user-idx", type=int, default=0, help="Which user_*.mp4 to use in the folder (0-based).")
    p.add_argument("--limit-folders", type=int, default=None, help="If --folder not set, run first N folders.")
    p.add_argument("--window-ratio", type=float, default=0.3, help="DTW window ratio (default 0.3).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(BASE_DIR)

    if args.folder:
        run_one_folder(args.folder, args.user_idx, args.window_ratio)
        return

    folders = discover_folders(args.limit_folders)
    print(f"Found {len(folders)} folders.")
    for f in folders:
        try:
            run_one_folder(f, args.user_idx, args.window_ratio)
        except Exception as e:
            print(f"[skip] {f}: {e}")


if __name__ == "__main__":
    main()

