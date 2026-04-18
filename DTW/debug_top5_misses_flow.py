"""
Visual debug flow for recognition misses (correct phrase not in Top-5).

What it does:
1) Loads recognition results JSON from evaluate_recognition.py output.
2) Selects cases where the correct folder is not in Top-5.
3) Plays 4-panel comparison with MediaPipe hand landmarks:
   - Panel 1: Correct translator video
   - Panel 2: User video
   - Panel 3: Wrong detected translator (rank-1 prediction)
   - Panel 4: Extra diagnostics (scores, frame ids, method stats)

Controls (display mode):
  Space: Pause / resume
  n:     Skip to next case
  q:     Quit
"""

from __future__ import annotations

import argparse
import json
import os
import bisect
import sys
import io
import re
import unicodedata
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import cosine as _cosine

# Fix Unicode output on Windows console (compare_three_methods prints symbols like "≥").
try:
    if getattr(sys.stdout, "encoding", "").lower() != "utf-8" and hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if getattr(sys.stderr, "encoding", "").lower() != "utf-8" and hasattr(sys.stderr, "buffer"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass


HEADER_H = 135
UI_FONT = cv2.FONT_HERSHEY_SIMPLEX
UI_COLOR_TEXT = (230, 230, 230)
UI_COLOR_MUTED = (180, 180, 180)
UI_COLOR_ACCENT = (80, 200, 255)
UI_COLOR_GOOD = (80, 220, 80)
UI_COLOR_BAD = (80, 80, 240)
UI_COLOR_WARN = (0, 200, 200)
UI_BG = (20, 20, 20)
PANEL_BORDER_CORRECT = (60, 220, 60)
PANEL_BORDER_USER = (255, 200, 80)
PANEL_BORDER_WRONG = (80, 80, 240)
PANEL_BORDER_INFO = (180, 180, 180)


def normalize_key(text: str) -> str:
    """Normalize folder labels so matching is robust to encoding differences."""
    ascii_text = (
        unicodedata.normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"[^a-z0-9]+", "", ascii_text.lower())


def safe_ascii(text: str, max_len: int = 72) -> str:
    """Return cv2-friendly overlay text."""
    cleaned = (
        unicodedata.normalize("NFKD", str(text))
        .encode("ascii", "ignore")
        .decode("ascii")
        .replace("\n", " ")
    ).strip()
    return cleaned if len(cleaned) <= max_len else cleaned[: max_len - 3] + "..."


def _draw_box(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    color: Tuple[int, int, int],
    alpha: float = 0.75,
) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def _put_line(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.55,
    color: Tuple[int, int, int] = UI_COLOR_TEXT,
    thickness: int = 1,
) -> None:
    cv2.putText(img, safe_ascii(text, 160), (x, y), UI_FONT, scale, color, thickness, cv2.LINE_AA)


def _fmt_score(x: Optional[float]) -> str:
    return "NA" if x is None else f"{x:.3f}"


def _score_color(x: Optional[float]) -> Tuple[int, int, int]:
    if x is None:
        return UI_COLOR_MUTED
    if x >= 0.75:
        return UI_COLOR_GOOD
    if x >= 0.45:
        return UI_COLOR_WARN
    return UI_COLOR_BAD


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except (TypeError, ValueError):
        return None


def _fit_canvas_for_display(canvas: np.ndarray, max_w: int = 1600, max_h: int = 900) -> np.ndarray:
    """
    Resize preview canvas to fit typical laptop screens.
    Keeps aspect ratio. Export/writer still uses original canvas size.
    """
    h, w = canvas.shape[:2]
    if h <= 0 or w <= 0:
        return canvas
    scale = min(max_w / float(w), max_h / float(h), 1.0)
    if scale >= 0.999:
        return canvas
    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))
    return cv2.resize(canvas, (nw, nh), interpolation=cv2.INTER_AREA)


class DiskCache:
    def __init__(self, root: Path, enabled: bool = True, recompute: bool = False):
        self.root = root
        self.enabled = enabled
        self.recompute = recompute
        if self.enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _hash(self, text: str) -> str:
        return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:20]

    def video_key(self, video_path: Path) -> str:
        p = video_path.resolve()
        try:
            st = p.stat()
            sig = f"{p}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            sig = f"{p}|missing"
        return self._hash(sig)

    def features_path(self, video_path: Path, similarity_threshold: float) -> Path:
        key = self.video_key(video_path)
        return self.root / f"features_{key}_thr{similarity_threshold:.3f}.npz"

    def align_path(self, kind: str, ref_video: Path, user_video: Path, params: str) -> Path:
        ref_k = self.video_key(ref_video)
        user_k = self.video_key(user_video)
        p = self._hash(params)
        return self.root / f"align_{kind}_{p}_{ref_k}_{user_k}.npz"


def load_results(results_json: Path) -> Dict:
    with results_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_top5_misses(detailed_rows: Sequence[Dict]) -> List[Dict]:
    misses: List[Dict] = []
    for row in detailed_rows:
        rank = int(row.get("correct_rank", 9999))
        in_top5 = bool(row.get("top5", rank <= 5))
        if (not in_top5) or rank > 5:
            misses.append(row)
    return misses


def build_folder_index(videos_dir: Path) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    exact: Dict[str, Path] = {}
    normalized: Dict[str, List[Path]] = {}

    for child in sorted(videos_dir.iterdir()):
        if not child.is_dir():
            continue
        exact[child.name] = child
        key = normalize_key(child.name)
        normalized.setdefault(key, []).append(child)

    return exact, normalized


def resolve_folder_path(
    folder_name: str,
    exact_index: Dict[str, Path],
    normalized_index: Dict[str, List[Path]],
) -> Optional[Path]:
    if folder_name in exact_index:
        return exact_index[folder_name]

    key = normalize_key(folder_name)
    matches = normalized_index.get(key, [])
    if len(matches) == 1:
        return matches[0]
    if matches:
        return matches[0]
    return None


def find_translator_video(folder_path: Path) -> Optional[Path]:
    candidates = sorted(folder_path.glob("translator_*.mp4"))
    return candidates[0] if candidates else None


def find_user_video(folder_path: Path, user_basename: str) -> Optional[Path]:
    direct = folder_path / user_basename
    if direct.exists():
        return direct

    target_lower = user_basename.lower()
    for candidate in folder_path.glob("user_*.mp4"):
        if candidate.name.lower() == target_lower:
            return candidate
    return None


def draw_landmarks_on_frame(
    frame_bgr: np.ndarray,
    hands_model,
    mp_hands,
    mp_draw,
    mp_styles,
) -> Tuple[np.ndarray, int, np.ndarray]:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = hands_model.process(frame_rgb)
    output = frame_bgr.copy()
    hands_detected = 0
    feat_vec = np.zeros((126,), dtype=np.float32)

    if results.multi_hand_landmarks:
        # Build a consistent 126-D feature vector: [left(63) | right(63)] with normalization.
        left = None
        right = None
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                output,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )
            hands_detected += 1

        try:
            # MediaPipe provides handedness aligned with multi_hand_landmarks
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                label = None
                if results.multi_handedness and idx < len(results.multi_handedness):
                    label = results.multi_handedness[idx].classification[0].label
                lm = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)  # (21,3)
                wrist = lm[0]
                centered = lm - wrist
                d = np.linalg.norm(centered, axis=1)
                max_d = float(np.max(d)) if d.size else 0.0
                if max_d > 0:
                    centered = centered / max_d
                if label == "Left":
                    left = centered
                elif label == "Right":
                    right = centered
                else:
                    # If handedness missing, store first in right then left.
                    if right is None:
                        right = centered
                    elif left is None:
                        left = centered

            left_feat = left.flatten() if left is not None else np.zeros((63,), dtype=np.float32)
            right_feat = right.flatten() if right is not None else np.zeros((63,), dtype=np.float32)
            feat_vec = np.concatenate([left_feat, right_feat]).astype(np.float32, copy=False)
        except Exception:
            pass
    return output, hands_detected, feat_vec


def fit_to_panel(frame_bgr: np.ndarray, panel_w: int, panel_h: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    scale = min(panel_w / w, panel_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    x0 = (panel_w - new_w) // 2
    y0 = (panel_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def top5_overlay_line(row: Dict, max_items: int = 3) -> str:
    ranking = row.get("top5_rankings", []) or []
    parts: List[str] = []
    for idx, item in enumerate(ranking[:max_items], start=1):
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            label = safe_ascii(item[0], max_len=24)
            try:
                score = float(item[1])
            except (TypeError, ValueError):
                score = 0.0
            parts.append(f"{idx}:{label} {score:.3f}")
    return " | ".join(parts)


def compose_view_four(
    correct_frame: np.ndarray,
    user_frame: np.ndarray,
    wrong_frame: np.ndarray,
    row: Dict,
    case_idx: int,
    total_cases: int,
    panel_w: int,
    panel_h: int,
    correct_title: str,
    user_title: str,
    wrong_title: str,
    correct_hands: int,
    user_hands: int,
    wrong_hands: int,
    info_title: str,
    info_lines: Sequence[str],
    sim_cu: Optional[float] = None,
    sim_wu: Optional[float] = None,
) -> np.ndarray:
    correct_panel = fit_to_panel(correct_frame, panel_w, panel_h)
    user_panel = fit_to_panel(user_frame, panel_w, panel_h)
    wrong_panel = fit_to_panel(wrong_frame, panel_w, panel_h)
    info_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    _draw_box(info_panel, 0, 0, panel_w - 1, panel_h - 1, (30, 30, 30), alpha=1.0)

    header_h = HEADER_H
    canvas = np.zeros((panel_h + header_h, panel_w * 4, 3), dtype=np.uint8)
    canvas[header_h:, :panel_w] = correct_panel
    canvas[header_h:, panel_w : panel_w * 2] = user_panel
    canvas[header_h:, panel_w * 2 : panel_w * 3] = wrong_panel
    canvas[header_h:, panel_w * 3 :] = info_panel

    # Panel borders (helps you know which is which at a glance).
    border = 4
    y0 = header_h
    y1 = header_h + panel_h - 1
    cv2.rectangle(canvas, (0, y0), (panel_w - 1, y1), PANEL_BORDER_CORRECT, border)
    cv2.rectangle(canvas, (panel_w, y0), (panel_w * 2 - 1, y1), PANEL_BORDER_USER, border)
    cv2.rectangle(canvas, (panel_w * 2, y0), (panel_w * 3 - 1, y1), PANEL_BORDER_WRONG, border)
    cv2.rectangle(canvas, (panel_w * 3, y0), (panel_w * 4 - 1, y1), PANEL_BORDER_INFO, border)

    correct_rank = int(row.get("correct_rank", 9999))
    top1_folder = safe_ascii(row.get("top1_folder", "?"), max_len=42)
    header1 = f"Case {case_idx}/{total_cases} | Correct rank: {correct_rank} | Top1: {top1_folder}"
    header2 = f"Correct: {safe_ascii(row.get('folder', '?'), 46)} | User: {safe_ascii(row.get('user_video', '?'), 42)}"
    # Keep header minimal; Top-5 list is noisy in this viewer.
    header3 = ""

    _put_line(canvas, header1, 10, 26, scale=0.62, color=UI_COLOR_TEXT, thickness=2)
    _put_line(canvas, header2, 10, 50, scale=0.54, color=UI_COLOR_TEXT, thickness=1)
    if header3:
        _put_line(canvas, header3, 10, 74, scale=0.50, color=UI_COLOR_MUTED, thickness=1)

    # Allow longer titles since we include frame indices + similarity + delta.
    correct_label = f"{safe_ascii(correct_title, 44)} | hands={correct_hands}"
    user_label = f"{safe_ascii(user_title, 44)} | hands={user_hands}"
    wrong_label = f"{safe_ascii(wrong_title, 44)} | hands={wrong_hands}"
    info_label = safe_ascii(info_title, 44)
    _put_line(canvas, correct_label, 12, header_h + 25, scale=0.55, color=UI_COLOR_GOOD, thickness=2)
    _put_line(canvas, user_label, panel_w + 12, header_h + 25, scale=0.55, color=UI_COLOR_GOOD, thickness=2)
    _put_line(canvas, wrong_label, panel_w * 2 + 12, header_h + 25, scale=0.55, color=UI_COLOR_GOOD, thickness=2)
    _put_line(canvas, info_label, panel_w * 3 + 12, header_h + 25, scale=0.55, color=UI_COLOR_TEXT, thickness=2)

    info_x0 = panel_w * 3 + 10
    info_y0 = header_h + 44
    info_w = panel_w - 20
    _draw_box(canvas, info_x0, info_y0, info_w, panel_h - 54, (35, 35, 40), alpha=0.95)

    y = info_y0 + 24
    if sim_cu is not None or sim_wu is not None:
        bar_w = info_w - 120
        _put_line(canvas, "C-U", info_x0 + 14, y, scale=0.52, color=_score_color(sim_cu), thickness=1)
        _draw_box(canvas, info_x0 + 65, y - 12, bar_w, 10, (55, 55, 65), alpha=0.9)
        if sim_cu is not None:
            _draw_box(canvas, info_x0 + 65, y - 12, int(bar_w * max(0.0, min(1.0, sim_cu))), 10, _score_color(sim_cu), alpha=0.95)
        _put_line(canvas, _fmt_score(sim_cu), info_x0 + 70 + bar_w, y, scale=0.48, color=UI_COLOR_TEXT, thickness=1)

        y += 20
        _put_line(canvas, "W-U", info_x0 + 14, y, scale=0.52, color=_score_color(sim_wu), thickness=1)
        _draw_box(canvas, info_x0 + 65, y - 12, bar_w, 10, (55, 55, 65), alpha=0.9)
        if sim_wu is not None:
            _draw_box(canvas, info_x0 + 65, y - 12, int(bar_w * max(0.0, min(1.0, sim_wu))), 10, _score_color(sim_wu), alpha=0.95)
        _put_line(canvas, _fmt_score(sim_wu), info_x0 + 70 + bar_w, y, scale=0.48, color=UI_COLOR_TEXT, thickness=1)
        y += 24

    for line in info_lines[:18]:
        _put_line(canvas, line, info_x0 + 14, y, scale=0.48, color=UI_COLOR_TEXT, thickness=1)
        y += 20

    return canvas


def open_writer(output_path: Path, fps: float, width: int, height: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    return writer


def read_frame_at_index(cap: cv2.VideoCapture, frame_idx: int) -> Optional[np.ndarray]:
    """Random-access read (0-based frame_idx)."""
    if frame_idx < 0:
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ret, frame = cap.read()
    return frame if (ret and frame is not None) else None


def _nearest_user_key(sorted_keys: List[int], target: int) -> Optional[int]:
    if not sorted_keys:
        return None
    pos = bisect.bisect_left(sorted_keys, target)
    if pos <= 0:
        return sorted_keys[0]
    if pos >= len(sorted_keys):
        return sorted_keys[-1]
    before = sorted_keys[pos - 1]
    after = sorted_keys[pos]
    return before if (target - before) <= (after - target) else after


def _build_user_to_ref_map(path: Sequence[Sequence[Any]]) -> Dict[int, Tuple[int, float]]:
    """Map user_frame -> (ref_frame, sim) choosing best sim if repeated."""
    out: Dict[int, Tuple[int, float]] = {}
    for item in path:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        ref_idx = int(item[0])
        user_idx = int(item[1])
        try:
            sim = float(item[2])
        except (TypeError, ValueError):
            sim = 0.0
        prev = out.get(user_idx)
        if prev is None or sim >= prev[1]:
            out[user_idx] = (ref_idx, sim)
    return out


def _pick_wrong_ref_for_user(
    user_idx: int,
    wrong_map: Dict[int, Tuple[int, float]],
    wrong_user_keys: List[int],
    fallback: str,
) -> Tuple[int, float, str]:
    """
    Returns:
      wrong_ref_idx, sim_wrong, mode
    mode is one of: "exact", "nearest", "none"
    """
    exact = wrong_map.get(user_idx)
    if exact is not None:
        return exact[0], exact[1], "exact"
    if fallback == "nearest":
        nearest_key = _nearest_user_key(wrong_user_keys, user_idx)
        if nearest_key is None:
            return 0, 0.0, "none"
        r_idx, s = wrong_map[nearest_key]
        return r_idx, s, "nearest"
    return 0, 0.0, "none"


def compute_offline_dtw_alignment(
    ref_video: Path,
    user_video: Path,
    window_ratio: float,
    similarity_threshold: float,
    feature_cache: Dict[str, Tuple[np.ndarray, List[int]]],
    disk_cache: Optional[DiskCache] = None,
) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    Compute DTW alignment path using the same functions as compare_three_methods.py.

    Returns:
        avg_similarity, path_with_original_indices: [(ref_frame_idx, user_frame_idx, sim), ...]
    """
    # Ensure matplotlib doesn't try to open a GUI backend if imported indirectly.
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Lazy import so normal playback mode stays lightweight.
    from compare_three_methods import dtw_align_with_prefiltered_frames

    if disk_cache is not None and disk_cache.enabled and not disk_cache.recompute:
        ap = disk_cache.align_path(
            "global",
            ref_video,
            user_video,
            params=f"thr={similarity_threshold:.3f}|win={window_ratio:.4f}",
        )
        if ap.exists():
            data = np.load(ap, allow_pickle=True)
            score = float(data["score"])
            path = data["path"].tolist()
            return score, [(int(a), int(b), float(s)) for (a, b, s) in path]

    ref_feat, ref_idx = _get_features_cached(ref_video, similarity_threshold, feature_cache, disk_cache)
    user_feat, user_idx = _get_features_cached(user_video, similarity_threshold, feature_cache, disk_cache)
    if len(ref_feat) == 0 or len(user_feat) == 0:
        return 0.0, []

    avg_sim, path, _frame_sims = dtw_align_with_prefiltered_frames(
        ref_feat, user_feat, ref_idx, user_idx, window_ratio=window_ratio
    )
    # path is already in original frame indices (see compare_three_methods.py)
    out_score = float(avg_sim)
    out_path = [(int(a), int(b), float(s)) for (a, b, s) in path]

    if disk_cache is not None and disk_cache.enabled:
        ap = disk_cache.align_path(
            "global",
            ref_video,
            user_video,
            params=f"thr={similarity_threshold:.3f}|win={window_ratio:.4f}",
        )
        try:
            np.savez_compressed(ap, score=np.array(out_score, dtype=np.float32), path=np.array(out_path, dtype=np.float32))
        except Exception:
            pass

    return out_score, out_path


def compute_offline_dtw_alignment_method6_cos_cost(
    ref_video: Path,
    user_video: Path,
    window_ratio: float,
    similarity_threshold: float,
    feature_cache: Dict[str, Tuple[np.ndarray, List[int]]],
    disk_cache: Optional[DiskCache] = None,
) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    Method 6: DTW alignment where the DP cost is cosine-distance (1 - cosine similarity).

    Returns:
        score, path_with_original_indices: [(ref_frame_idx, user_frame_idx, cos_sim), ...]
    """
    os.environ.setdefault("MPLBACKEND", "Agg")

    if disk_cache is not None and disk_cache.enabled and not disk_cache.recompute:
        ap = disk_cache.align_path(
            "m6",
            ref_video,
            user_video,
            params=f"thr={similarity_threshold:.3f}|win={window_ratio:.4f}",
        )
        if ap.exists():
            data = np.load(ap, allow_pickle=True)
            score = float(data["score"])
            path = data["path"].tolist()
            return score, [(int(a), int(b), float(s)) for (a, b, s) in path]

    ref_feat, ref_idx = _get_features_cached(ref_video, similarity_threshold, feature_cache, disk_cache)
    user_feat, user_idx = _get_features_cached(user_video, similarity_threshold, feature_cache, disk_cache)
    if len(ref_feat) == 0 or len(user_feat) == 0:
        return 0.0, []

    from method6_dtw_cos_cost import dtw_align_cosine_cost

    r = dtw_align_cosine_cost(
        ref_feat,
        user_feat,
        ref_idx,
        user_idx,
        window_ratio=window_ratio,
    )
    out_score = float(r.score)
    out_path = [(int(a), int(b), float(s)) for (a, b, s) in r.path]

    if disk_cache is not None and disk_cache.enabled:
        ap = disk_cache.align_path(
            "m6",
            ref_video,
            user_video,
            params=f"thr={similarity_threshold:.3f}|win={window_ratio:.4f}",
        )
        try:
            np.savez_compressed(ap, score=np.array(out_score, dtype=np.float32), path=np.array(out_path, dtype=np.float32))
        except Exception:
            pass

    return out_score, out_path


def compute_offline_dtw_alignment_method4_variant(
    ref_video: Path,
    user_video: Path,
    *,
    feature_kind: str,
    swap_mode: str,
    window_ratio: float,
    similarity_threshold: float,
    feature_cache: Dict[str, Tuple[np.ndarray, List[int]]],
    disk_cache: Optional[DiskCache] = None,
) -> Tuple[float, List[Tuple[int, int, float]], bool]:
    """
    Method4-variant alignment (bones/angles features) using method4_variants.dtw_global_align.

    Returns:
        score, path_with_original_indices, swapped_used
    """
    os.environ.setdefault("MPLBACKEND", "Agg")

    if disk_cache is not None and disk_cache.enabled and not disk_cache.recompute:
        ap = disk_cache.align_path(
            "m4v",
            ref_video,
            user_video,
            params=f"kind={feature_kind}|swap={swap_mode}|thr={similarity_threshold:.3f}|win={window_ratio:.4f}",
        )
        if ap.exists():
            data = np.load(ap, allow_pickle=True)
            score = float(data["score"])
            path = data["path"].tolist()
            swapped_used = bool(int(data["swapped_used"])) if ("swapped_used" in data.files) else False
            return score, [(int(a), int(b), float(s)) for (a, b, s) in path], swapped_used

    ref_coords, ref_idx = _get_features_cached(ref_video, similarity_threshold, feature_cache, disk_cache)
    user_coords, user_idx = _get_features_cached(user_video, similarity_threshold, feature_cache, disk_cache)
    if len(ref_coords) == 0 or len(user_coords) == 0:
        return 0.0, [], False

    from method4_variants import (
        dtw_global_align,
        features_from_coords_matrix,
    )

    ref_feat = features_from_coords_matrix(ref_coords, feature_kind)  # type: ignore[arg-type]
    user_feat = features_from_coords_matrix(user_coords, feature_kind)  # type: ignore[arg-type]
    r = dtw_global_align(
        ref_feat,
        user_feat,
        ref_idx,
        user_idx,
        window_ratio=window_ratio,
        kind=feature_kind,      # type: ignore[arg-type]
        swap_mode=swap_mode,    # type: ignore[arg-type]
    )

    out_score = float(r.score)
    out_path = [(int(a), int(b), float(s)) for (a, b, s) in r.path]
    out_swapped = bool(r.swapped_used)

    if disk_cache is not None and disk_cache.enabled:
        ap = disk_cache.align_path(
            "m4v",
            ref_video,
            user_video,
            params=f"kind={feature_kind}|swap={swap_mode}|thr={similarity_threshold:.3f}|win={window_ratio:.4f}",
        )
        try:
            np.savez_compressed(
                ap,
                score=np.array(out_score, dtype=np.float32),
                path=np.array(out_path, dtype=np.float32),
                swapped_used=np.array(1 if out_swapped else 0, dtype=np.int8),
            )
        except Exception:
            pass

    return out_score, out_path, out_swapped


def _build_feat_lookup_method4_variant(
    video_path: Path,
    *,
    feature_kind: str,
    similarity_threshold: float,
    feature_cache: Dict[str, Tuple[np.ndarray, List[int]]],
    lookup_cache: Dict[str, Dict[int, np.ndarray]],
    disk_cache: Optional[DiskCache] = None,
) -> Dict[int, np.ndarray]:
    """
    Map original_frame_idx -> method4-variant feature vector for kept frames.
    Cached in-memory per (video_path,feature_kind).
    """
    key = f"{str(video_path.resolve())}|m4v|{feature_kind}"
    cached = lookup_cache.get(key)
    if cached is not None:
        return cached

    coords, idx = _get_features_cached(video_path, similarity_threshold, feature_cache, disk_cache)
    from method4_variants import features_from_coords_matrix

    feat = features_from_coords_matrix(coords, feature_kind)  # type: ignore[arg-type]
    out: Dict[int, np.ndarray] = {}
    for i, orig in enumerate(idx):
        out[int(orig)] = feat[i]
    lookup_cache[key] = out
    return out


def compute_offline_dtw_alignment_method5_user_swap(
    ref_video: Path,
    user_video: Path,
    window_ratio: float,
    similarity_threshold: float,
    feature_cache: Dict[str, Tuple[np.ndarray, List[int]]],
    disk_cache: Optional[DiskCache] = None,
) -> Tuple[float, List[Tuple[int, int, float]], str]:
    """
    Method 5: global DTW, but pick best of (orig user features) vs (user features with L/R halves swapped).

    Returns:
        score, path_with_original_indices, variant ("orig" or "swap")
    """
    os.environ.setdefault("MPLBACKEND", "Agg")

    if disk_cache is not None and disk_cache.enabled and not disk_cache.recompute:
        ap = disk_cache.align_path(
            "m5",
            ref_video,
            user_video,
            params=f"thr={similarity_threshold:.3f}|win={window_ratio:.4f}",
        )
        if ap.exists():
            data = np.load(ap, allow_pickle=True)
            score = float(data["score"])
            path = data["path"].tolist()
            variant = str(data["variant"]) if ("variant" in data.files) else "orig"
            return score, [(int(a), int(b), float(s)) for (a, b, s) in path], variant

    ref_feat, ref_idx = _get_features_cached(ref_video, similarity_threshold, feature_cache, disk_cache)
    user_feat, user_idx = _get_features_cached(user_video, similarity_threshold, feature_cache, disk_cache)
    if len(ref_feat) == 0 or len(user_feat) == 0:
        return 0.0, [], "orig"

    from method5_dtw_cos_swap import dtw_best_of_user_swap

    r = dtw_best_of_user_swap(
        ref_feat,
        user_feat,
        ref_idx,
        user_idx,
        window_ratio=window_ratio,
        verbose=False,
    )
    out_score = float(r.score)
    out_path = [(int(a), int(b), float(s)) for (a, b, s) in r.path]
    out_variant = str(r.variant)

    if disk_cache is not None and disk_cache.enabled:
        ap = disk_cache.align_path(
            "m5",
            ref_video,
            user_video,
            params=f"thr={similarity_threshold:.3f}|win={window_ratio:.4f}",
        )
        try:
            np.savez_compressed(
                ap,
                score=np.array(out_score, dtype=np.float32),
                path=np.array(out_path, dtype=np.float32),
                variant=np.array(out_variant),
            )
        except Exception:
            pass

    return out_score, out_path, out_variant


def compute_subsequence_dtw_alignment(
    ref_video: Path,
    user_video: Path,
    window_ratio: float,
    similarity_threshold: float,
    feature_cache: Dict[str, Tuple[np.ndarray, List[int]]],
    disk_cache: Optional[DiskCache] = None,
) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    Camera-like subsequence DTW (same algorithm family as streaming_dtw.offline_full_dtw).

    Uses:
      - Euclidean cost matrix with diagonal-centered Sakoe-Chiba band
      - Subsequence accumulated cost (free start in user)
      - Backtrack from best end column
      - Local optimization to maximize cosine similarity per aligned pair

    Returns:
        avg_similarity, path_with_original_indices: [(ref_frame_idx, user_frame_idx, sim), ...]
    """
    from streaming_dtw import (
        compute_cost_matrix,
        subsequence_dtw_accumulated,
        subsequence_dtw_backtrack,
        MAX_USER_FRAME_REUSE,
    )

    if disk_cache is not None and disk_cache.enabled and not disk_cache.recompute:
        ap = disk_cache.align_path(
            "subseq",
            ref_video,
            user_video,
            params=f"thr={similarity_threshold:.3f}|win={window_ratio:.4f}",
        )
        if ap.exists():
            data = np.load(ap, allow_pickle=True)
            score = float(data["score"])
            path = data["path"].tolist()
            return score, [(int(a), int(b), float(s)) for (a, b, s) in path]

    ref_feat, ref_idx = _get_features_cached(ref_video, similarity_threshold, feature_cache, disk_cache)
    user_feat, user_idx = _get_features_cached(user_video, similarity_threshold, feature_cache, disk_cache)
    N = len(ref_feat)
    M = len(user_feat)
    if N == 0 or M == 0:
        return 0.0, []

    # Subsequence DTW uses a diagonal-centered band in compute_cost_matrix.
    C = compute_cost_matrix(ref_feat, user_feat, window_ratio=window_ratio)
    D = subsequence_dtw_accumulated(C)
    path_arr, _a_star, _b_star = subsequence_dtw_backtrack(D)
    if path_arr.size == 0:
        return 0.0, []

    # Window for local cosine optimization (keep consistent with streaming_dtw defaults).
    opt_window = max(1, int(window_ratio * max(N, M)))

    frame_sims: List[float] = []
    path_with_indices: List[Tuple[int, int, float]] = []
    last_chosen_m = 0
    usage_count: Dict[int, int] = {}

    for pidx in range(len(path_arr)):
        n_idx = int(path_arr[pidx, 0])
        m_idx = int(path_arr[pidx, 1])
        vec_r = ref_feat[n_idx]

        if pidx > 0:
            min_m = int(path_arr[pidx - 1, 1])
        else:
            min_m = max(0, m_idx - opt_window)

        if pidx < len(path_arr) - 1:
            max_m = int(path_arr[pidx + 1, 1])
        else:
            max_m = min(M, m_idx + opt_window + 1)

        min_m = max(min_m, last_chosen_m)

        best_sim = -1.0
        best_m = -1

        for alt_m in range(min_m, max_m):
            if usage_count.get(alt_m, 0) >= MAX_USER_FRAME_REUSE:
                continue
            vec_u = user_feat[alt_m]
            nu = float(np.linalg.norm(vec_u))
            if nu <= 0:
                continue
            alt_sim = float(1.0 - _cosine(vec_r, vec_u))
            if alt_sim > best_sim:
                best_sim = alt_sim
                best_m = alt_m

        if best_m < 0:
            best_m = last_chosen_m
            best_sim = 0.0

        last_chosen_m = best_m
        usage_count[best_m] = usage_count.get(best_m, 0) + 1

        frame_sims.append(best_sim)
        orig_r = int(ref_idx[n_idx])
        orig_u = int(user_idx[best_m])
        path_with_indices.append((orig_r, orig_u, float(best_sim)))

    out_score = float(np.mean(frame_sims)) if frame_sims else 0.0
    out_path = path_with_indices

    if disk_cache is not None and disk_cache.enabled:
        ap = disk_cache.align_path(
            "subseq",
            ref_video,
            user_video,
            params=f"thr={similarity_threshold:.3f}|win={window_ratio:.4f}",
        )
        try:
            np.savez_compressed(ap, score=np.array(out_score, dtype=np.float32), path=np.array(out_path, dtype=np.float32))
        except Exception:
            pass

    return out_score, out_path


def _get_features_cached(
    video_path: Path,
    similarity_threshold: float,
    feature_cache: Dict[str, Tuple[np.ndarray, List[int]]],
    disk_cache: Optional[DiskCache] = None,
) -> Tuple[np.ndarray, List[int]]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    from compare_three_methods import extract_all_frames_and_filter

    key = str(video_path.resolve())
    cached = feature_cache.get(key)
    if cached is not None:
        return cached

    if disk_cache is not None and disk_cache.enabled and not disk_cache.recompute:
        fp = disk_cache.features_path(video_path, similarity_threshold)
        if fp.exists():
            data = np.load(fp, allow_pickle=True)
            feat = data["features"]
            idx = data["indices"].tolist()
            feature_cache[key] = (feat, idx)
            return feat, idx

    feat, idx = extract_all_frames_and_filter(str(video_path), similarity_threshold=similarity_threshold)
    feature_cache[key] = (feat, idx)

    if disk_cache is not None and disk_cache.enabled:
        fp = disk_cache.features_path(video_path, similarity_threshold)
        try:
            np.savez_compressed(fp, features=feat, indices=np.array(idx, dtype=np.int32))
        except Exception:
            pass

    return feat, idx


def compute_index_pair_alignment(
    ref_video: Path,
    user_video: Path,
    similarity_threshold: float,
    feature_cache: Dict[str, Tuple[np.ndarray, List[int]]],
    disk_cache: Optional[DiskCache] = None,
) -> Tuple[float, List[Tuple[int, int, float]]]:
    """
    Index-based alignment (no DTW): kept frame i <-> kept frame i.
    Returns pairs using original frame indices.
    """
    ref_feat, ref_idx = _get_features_cached(ref_video, similarity_threshold, feature_cache, disk_cache)
    user_feat, user_idx = _get_features_cached(user_video, similarity_threshold, feature_cache, disk_cache)
    n = min(len(ref_feat), len(user_feat))
    if n <= 0:
        return 0.0, []

    sims: List[float] = []
    pairs: List[Tuple[int, int, float]] = []
    for i in range(n):
        v1 = ref_feat[i]
        v2 = user_feat[i]
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        sim = float(1.0 - _cosine(v1, v2)) if (n1 > 0 and n2 > 0) else 0.0
        sims.append(sim)
        pairs.append((int(ref_idx[i]), int(user_idx[i]), sim))
    return float(np.mean(sims)), pairs


def compute_method1_flatten_score(
    ref_video: Path,
    user_video: Path,
    similarity_threshold: float,
    feature_cache: Dict[str, Tuple[np.ndarray, List[int]]],
    disk_cache: Optional[DiskCache] = None,
) -> float:
    """
    Method 1 score: flatten kept frames (truncate to same kept length) then cosine.
    """
    ref_feat, _ = _get_features_cached(ref_video, similarity_threshold, feature_cache, disk_cache)
    user_feat, _ = _get_features_cached(user_video, similarity_threshold, feature_cache, disk_cache)
    n = min(len(ref_feat), len(user_feat))
    if n <= 0:
        return 0.0

    v1 = ref_feat[:n].flatten()
    v2 = user_feat[:n].flatten()
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    return float(1.0 - _cosine(v1, v2)) if (n1 > 0 and n2 > 0) else 0.0


def _build_feat_lookup(
    video_path: Path,
    similarity_threshold: float,
    feature_cache: Dict[str, Tuple[np.ndarray, List[int]]],
    lookup_cache: Dict[str, Dict[int, np.ndarray]],
    disk_cache: Optional[DiskCache] = None,
) -> Dict[int, np.ndarray]:
    """Map original_frame_idx -> feature_vector for kept frames."""
    key = str(video_path.resolve())
    cached = lookup_cache.get(key)
    if cached is not None:
        return cached

    feat, idx = _get_features_cached(video_path, similarity_threshold, feature_cache, disk_cache)
    out: Dict[int, np.ndarray] = {}
    for i, orig in enumerate(idx):
        out[int(orig)] = feat[i]
    lookup_cache[key] = out
    return out


def _cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    n1 = float(np.linalg.norm(a))
    n2 = float(np.linalg.norm(b))
    if n1 <= 0 or n2 <= 0:
        return 0.0
    return float(1.0 - _cosine(a, b))


def _m4_component_sims(
    vec_c: Optional[np.ndarray],
    vec_u: Optional[np.ndarray],
    vec_w: Optional[np.ndarray],
    kind: str,
) -> Dict[str, Optional[float]]:
    """
    For Method4-variant frame features, compute compact component similarities:
      - angles: cosine on angle subvector(s)
      - lengths: cosine on length subvector(s)
    """
    out: Dict[str, Optional[float]] = {
        "ang_cu": None,
        "ang_wu": None,
        "len_cu": None,
        "len_wu": None,
    }
    if vec_u is None:
        return out

    # bones_angles_len_d1: [base420 | d1420], base420 = [left(190A+20L) | right(190A+20L)]
    if kind == "bones_angles_len_d1":
        if vec_c is not None and vec_u is not None and vec_c.shape[0] >= 420 and vec_u.shape[0] >= 420:
            c_base = vec_c[:420]
            u_base = vec_u[:420]
            c_ang = np.concatenate([c_base[:190], c_base[210:400]])
            u_ang = np.concatenate([u_base[:190], u_base[210:400]])
            c_len = np.concatenate([c_base[190:210], c_base[400:420]])
            u_len = np.concatenate([u_base[190:210], u_base[400:420]])
            out["ang_cu"] = _cosine_similarity(c_ang, u_ang)
            out["len_cu"] = _cosine_similarity(c_len, u_len)

        if vec_w is not None and vec_u is not None and vec_w.shape[0] >= 420 and vec_u.shape[0] >= 420:
            w_base = vec_w[:420]
            u_base = vec_u[:420]
            w_ang = np.concatenate([w_base[:190], w_base[210:400]])
            u_ang = np.concatenate([u_base[:190], u_base[210:400]])
            w_len = np.concatenate([w_base[190:210], w_base[400:420]])
            u_len = np.concatenate([u_base[190:210], u_base[400:420]])
            out["ang_wu"] = _cosine_similarity(w_ang, u_ang)
            out["len_wu"] = _cosine_similarity(w_len, u_len)
        return out

    # bones_angles_len: base420 only
    if kind == "bones_angles_len":
        if vec_c is not None and vec_u is not None and vec_c.shape[0] >= 420 and vec_u.shape[0] >= 420:
            c_ang = np.concatenate([vec_c[:190], vec_c[210:400]])
            u_ang = np.concatenate([vec_u[:190], vec_u[210:400]])
            c_len = np.concatenate([vec_c[190:210], vec_c[400:420]])
            u_len = np.concatenate([vec_u[190:210], vec_u[400:420]])
            out["ang_cu"] = _cosine_similarity(c_ang, u_ang)
            out["len_cu"] = _cosine_similarity(c_len, u_len)

        if vec_w is not None and vec_u is not None and vec_w.shape[0] >= 420 and vec_u.shape[0] >= 420:
            w_ang = np.concatenate([vec_w[:190], vec_w[210:400]])
            u_ang = np.concatenate([vec_u[:190], vec_u[210:400]])
            w_len = np.concatenate([vec_w[190:210], vec_w[400:420]])
            u_len = np.concatenate([vec_u[190:210], vec_u[400:420]])
            out["ang_wu"] = _cosine_similarity(w_ang, u_ang)
            out["len_wu"] = _cosine_similarity(w_len, u_len)
        return out

    return out


def play_case_three(
    row: Dict,
    case_idx: int,
    total_cases: int,
    correct_video: Path,
    user_video: Path,
    wrong_video: Path,
    correct_title: str,
    user_title: str,
    wrong_title: str,
    panel_w: int,
    panel_h: int,
    fps_override: float,
    display: bool,
    save_path: Optional[Path],
    mode: str,
    method_view: str,
    dtw_window_ratio: float,
    subseq_window_ratio: float,
    similarity_threshold: float,
    feature_cache: Dict[str, Tuple[np.ndarray, List[int]]],
    disk_cache: Optional[DiskCache],
    max_pairs: int,
    wrong_fallback: str,
) -> str:
    cap_correct = cv2.VideoCapture(str(correct_video))
    cap_user = cv2.VideoCapture(str(user_video))
    cap_wrong = cv2.VideoCapture(str(wrong_video))
    if (not cap_correct.isOpened()) or (not cap_user.isOpened()) or (not cap_wrong.isOpened()):
        print(f"  [skip] Could not open video(s): {correct_video.name}, {user_video.name}, {wrong_video.name}")
        return "next"

    correct_fps = cap_correct.get(cv2.CAP_PROP_FPS)
    user_fps = cap_user.get(cv2.CAP_PROP_FPS)
    wrong_fps = cap_wrong.get(cv2.CAP_PROP_FPS)
    measured = [f for f in (correct_fps, user_fps, wrong_fps) if f and f >= 1.0]
    fps = fps_override if fps_override > 0 else (min(measured) if measured else 25.0)
    wait_ms = max(1, int(1000 / fps))

    canvas_w = panel_w * 4
    canvas_h = panel_h + HEADER_H
    writer = open_writer(save_path, fps, canvas_w, canvas_h) if save_path else None

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles
    hands_correct = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands_user = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hands_wrong = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    paused = False  # time mode pause
    match_autoplay = True  # match mode autoplay (user can toggle with Space)
    help_visible = True
    match_data: Optional[Dict[str, Any]] = None
    feat_lookup_cache: Dict[str, Dict[int, np.ndarray]] = {}
    match_idx = 0
    frame_idx = 0
    frames_written = 0
    blank = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    latest_canvas = compose_view_four(
        blank,
        blank,
        blank,
        row,
        case_idx,
        total_cases,
        panel_w,
        panel_h,
        correct_title,
        user_title,
        wrong_title,
        0,
        0,
        0,
        "Extra Info",
        ["Waiting for frames..."],
        None,
        None,
    )

    try:
        while True:
            if mode == "time":
                if not paused:
                    ret_correct, frame_correct = cap_correct.read()
                    ret_user, frame_user = cap_user.read()
                    ret_wrong, frame_wrong = cap_wrong.read()
                    if not ret_correct and not ret_user and not ret_wrong:
                        break
                    if not ret_correct:
                        frame_correct = np.zeros((480, 640, 3), dtype=np.uint8)
                    if not ret_user:
                        frame_user = np.zeros((480, 640, 3), dtype=np.uint8)
                    if not ret_wrong:
                        frame_wrong = np.zeros((480, 640, 3), dtype=np.uint8)

                    frame_correct, n_hands_correct, feat_c = draw_landmarks_on_frame(frame_correct, hands_correct, mp_hands, mp_draw, mp_styles)
                    frame_user, n_hands_user, feat_u = draw_landmarks_on_frame(frame_user, hands_user, mp_hands, mp_draw, mp_styles)
                    frame_wrong, n_hands_wrong, feat_w = draw_landmarks_on_frame(frame_wrong, hands_wrong, mp_hands, mp_draw, mp_styles)

                    sim_cu = _cosine_similarity(feat_c, feat_u)
                    sim_wu = _cosine_similarity(feat_w, feat_u)
                    sim_cw = _cosine_similarity(feat_c, feat_w)
                    delta_live = (sim_wu - sim_cu) if (sim_wu is not None and sim_cu is not None) else None
                    info_lines = [
                        f"Mode: TIME (sync playback)",
                        f"Frame idx (sync): {frame_idx}",
                        f"C-U: {_fmt_score(sim_cu)}",
                        f"W-U: {_fmt_score(sim_wu)}",
                        f"Delta (W-U - C-U): {_fmt_score(delta_live)}",
                        f"C-W: {_fmt_score(sim_cw)}",
                        f"Hands C/U/W: {n_hands_correct}/{n_hands_user}/{n_hands_wrong}",
                        "Use T to switch to MATCH mode.",
                    ]
                    latest_canvas = compose_view_four(
                        frame_correct,
                        frame_user,
                        frame_wrong,
                        row,
                        case_idx,
                        total_cases,
                        panel_w,
                        panel_h,
                        correct_title,
                        user_title,
                        wrong_title,
                        n_hands_correct,
                        n_hands_user,
                        n_hands_wrong,
                        "Extra Info",
                        info_lines,
                        sim_cu,
                        sim_wu,
                    )
                    frame_idx += 1
                    if writer is not None:
                        writer.write(latest_canvas)
                        frames_written += 1

                    _put_line(latest_canvas, "Keys: T match-mode | Space pause | H help | N next | Q quit", 18, latest_canvas.shape[0] - 12, scale=0.48, color=UI_COLOR_MUTED, thickness=1)

                if display:
                    preview = _fit_canvas_for_display(latest_canvas)
                    cv2.imshow("Top-5 Miss Debug Flow (Correct | User | Wrong | Extra Info)", preview)
                    key = cv2.waitKey(0 if paused else wait_ms) & 0xFF
                    if key == ord("q"):
                        return "quit"
                    if key == ord("n"):
                        return "next"
                    if key == ord(" "):
                        paused = not paused
                    if key == ord("m") or key == ord("t") or key == ord("T"):
                        mode = "match"
                    if key == ord("h") or key == ord("H") or key == ord("?"):
                        help_visible = not help_visible
                else:
                    continue

            else:
                # match mode: show per-method matched frame pairs with similarities
                if match_data is None:
                    print("  Precomputing alignments for Method4/6 only...")
                    # Method 4 variant (best full-run config we have): bones_angles_len_d1 + global swap + win=0.25
                    m4_kind = match_data.get("m4_kind", "bones_angles_len_d1") if match_data else "bones_angles_len_d1"
                    m4_swap = match_data.get("m4_swap", "global") if match_data else "global"
                    m4_win = float(match_data.get("m4_win", 0.25)) if match_data else 0.25
                    print("  Computing DTW alignment (Method4 variant) correct vs user...")
                    m4_correct_score, m4_pairs, m4_correct_swapped = compute_offline_dtw_alignment_method4_variant(
                        correct_video, user_video,
                        feature_kind=m4_kind,
                        swap_mode=m4_swap,
                        window_ratio=m4_win,
                        similarity_threshold=similarity_threshold,
                        feature_cache=feature_cache,
                        disk_cache=disk_cache,
                    )
                    print("  Computing DTW alignment (Method4 variant) wrong(top1) vs user...")
                    m4_wrong_score, m4_wrong_pairs, m4_wrong_swapped = compute_offline_dtw_alignment_method4_variant(
                        wrong_video, user_video,
                        feature_kind=m4_kind,
                        swap_mode=m4_swap,
                        window_ratio=m4_win,
                        similarity_threshold=similarity_threshold,
                        feature_cache=feature_cache,
                        disk_cache=disk_cache,
                    )
                    # Method 6: DTW cosine-cost (global)
                    print("  Computing DTW alignment (Method6 cosine-cost) correct vs user...")
                    m6_correct_score, m6_pairs = compute_offline_dtw_alignment_method6_cos_cost(
                        correct_video, user_video,
                        window_ratio=dtw_window_ratio,
                        similarity_threshold=similarity_threshold,
                        feature_cache=feature_cache,
                        disk_cache=disk_cache,
                    )
                    print("  Computing DTW alignment (Method6 cosine-cost) wrong(top1) vs user...")
                    m6_wrong_score, m6_wrong_pairs = compute_offline_dtw_alignment_method6_cos_cost(
                        wrong_video, user_video,
                        window_ratio=dtw_window_ratio,
                        similarity_threshold=similarity_threshold,
                        feature_cache=feature_cache,
                        disk_cache=disk_cache,
                    )
                    m6_wrong_map = _build_user_to_ref_map(m6_wrong_pairs)
                    m6_wrong_user_keys = sorted(m6_wrong_map.keys())
                    m4_wrong_map = _build_user_to_ref_map(m4_wrong_pairs)
                    m4_wrong_user_keys = sorted(m4_wrong_map.keys())
                    m4_correct_lookup = _build_feat_lookup_method4_variant(
                        correct_video,
                        feature_kind=m4_kind,
                        similarity_threshold=similarity_threshold,
                        feature_cache=feature_cache,
                        lookup_cache=feat_lookup_cache,
                        disk_cache=disk_cache,
                    )
                    m4_user_lookup = _build_feat_lookup_method4_variant(
                        user_video,
                        feature_kind=m4_kind,
                        similarity_threshold=similarity_threshold,
                        feature_cache=feature_cache,
                        lookup_cache=feat_lookup_cache,
                        disk_cache=disk_cache,
                    )
                    m4_wrong_lookup = _build_feat_lookup_method4_variant(
                        wrong_video,
                        feature_kind=m4_kind,
                        similarity_threshold=similarity_threshold,
                        feature_cache=feature_cache,
                        lookup_cache=feat_lookup_cache,
                        disk_cache=disk_cache,
                    )
                    correct_lookup = _build_feat_lookup(correct_video, similarity_threshold, feature_cache, feat_lookup_cache, disk_cache)
                    user_lookup = _build_feat_lookup(user_video, similarity_threshold, feature_cache, feat_lookup_cache, disk_cache)
                    wrong_lookup = _build_feat_lookup(wrong_video, similarity_threshold, feature_cache, feat_lookup_cache, disk_cache)
                    match_data = {
                        "m4_kind": m4_kind,
                        "m4_swap": m4_swap,
                        "m4_win": m4_win,
                        "m4_correct_score": m4_correct_score,
                        "m4_wrong_score": m4_wrong_score,
                        "m4_pairs": m4_pairs,
                        "m4_wrong_map": m4_wrong_map,
                        "m4_wrong_user_keys": m4_wrong_user_keys,
                        "m4_correct_swapped": m4_correct_swapped,
                        "m4_wrong_swapped": m4_wrong_swapped,
                        "m6_correct_score": m6_correct_score,
                        "m6_wrong_score": m6_wrong_score,
                        "m6_pairs": m6_pairs,
                        "m6_wrong_map": m6_wrong_map,
                        "m6_wrong_user_keys": m6_wrong_user_keys,
                        "correct_lookup": correct_lookup,
                        "user_lookup": user_lookup,
                        "wrong_lookup": wrong_lookup,
                        "m4_correct_lookup": m4_correct_lookup,
                        "m4_user_lookup": m4_user_lookup,
                        "m4_wrong_lookup": m4_wrong_lookup,
                    }
                    match_idx = 0

                view = method_view
                limit_pairs = max_pairs if max_pairs and max_pairs > 0 else 0

                if view == "m6":
                    pairs_correct = match_data["m6_pairs"]
                    wrong_map = match_data["m6_wrong_map"]
                    wrong_user_keys = match_data["m6_wrong_user_keys"]
                    if not pairs_correct:
                        print("  [skip] No DTW path produced for Method6 correct vs user")
                        return "next"
                    pair_count = len(pairs_correct)
                    limit_pairs = limit_pairs if limit_pairs > 0 else pair_count
                    match_idx = max(0, min(match_idx, limit_pairs - 1))
                    ref_idx, usr_idx, sim_correct = pairs_correct[match_idx]
                    wrong_ref_idx, sim_wrong, wrong_mode = _pick_wrong_ref_for_user(
                        usr_idx, wrong_map, wrong_user_keys, fallback=wrong_fallback
                    )
                    delta = (sim_wrong - sim_correct) if wrong_mode != "none" else None
                    method_label = f"M6(CosCostDTW): correct={match_data['m6_correct_score']:.3f} wrong={match_data['m6_wrong_score']:.3f}"
                elif view == "m4":
                    pairs_correct = match_data["m4_pairs"]
                    wrong_map = match_data["m4_wrong_map"]
                    wrong_user_keys = match_data["m4_wrong_user_keys"]
                    if not pairs_correct:
                        print("  [skip] No DTW path produced for Method4-variant correct vs user")
                        return "next"
                    pair_count = len(pairs_correct)
                    limit_pairs = limit_pairs if limit_pairs > 0 else pair_count
                    match_idx = max(0, min(match_idx, limit_pairs - 1))
                    ref_idx, usr_idx, sim_correct = pairs_correct[match_idx]
                    wrong_ref_idx, sim_wrong, wrong_mode = _pick_wrong_ref_for_user(
                        usr_idx, wrong_map, wrong_user_keys, fallback=wrong_fallback
                    )
                    delta = (sim_wrong - sim_correct) if wrong_mode != "none" else None
                    kind = match_data.get("m4_kind", "bones_angles_len_d1")
                    win = match_data.get("m4_win", 0.25)
                    swap = match_data.get("m4_swap", "global")
                    cs = "swap" if match_data.get("m4_correct_swapped", False) else "orig"
                    ws = "swap" if match_data.get("m4_wrong_swapped", False) else "orig"
                    method_label = f"M4V({kind}, swap={swap}, win={win:.2f}): correct={match_data['m4_correct_score']:.3f} [{cs}] wrong={match_data['m4_wrong_score']:.3f} [{ws}]"
                else:
                    # Default to Method4 if an unexpected view value appears.
                    method_view = "m4"
                    continue

                if match_idx >= limit_pairs:
                    break

                frame_correct = read_frame_at_index(cap_correct, ref_idx)
                if frame_correct is None:
                    frame_correct = np.zeros((480, 640, 3), dtype=np.uint8)

                frame_user = read_frame_at_index(cap_user, usr_idx)
                if frame_user is None:
                    frame_user = np.zeros((480, 640, 3), dtype=np.uint8)

                if wrong_mode == "none":
                    frame_wrong = np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    frame_wrong = read_frame_at_index(cap_wrong, wrong_ref_idx)
                    if frame_wrong is None:
                        frame_wrong = np.zeros((480, 640, 3), dtype=np.uint8)

                frame_correct, n_hands_correct, _ = draw_landmarks_on_frame(frame_correct, hands_correct, mp_hands, mp_draw, mp_styles)
                frame_user, n_hands_user, _ = draw_landmarks_on_frame(frame_user, hands_user, mp_hands, mp_draw, mp_styles)
                frame_wrong, n_hands_wrong, _ = draw_landmarks_on_frame(frame_wrong, hands_wrong, mp_hands, mp_draw, mp_styles)

                # extra diagnostics for match mode
                correct_lookup = match_data.get("correct_lookup", {})
                user_lookup = match_data.get("user_lookup", {})
                wrong_lookup = match_data.get("wrong_lookup", {})
                if view == "m4":
                    correct_lookup = match_data.get("m4_correct_lookup", {})
                    user_lookup = match_data.get("m4_user_lookup", {})
                    wrong_lookup = match_data.get("m4_wrong_lookup", {})
                vec_c = correct_lookup.get(int(ref_idx))
                vec_u = user_lookup.get(int(usr_idx))
                vec_w = None if wrong_mode == "none" else wrong_lookup.get(int(wrong_ref_idx))

                sim_cu = _cosine_similarity(vec_c, vec_u)
                sim_wu = _cosine_similarity(vec_w, vec_u)
                sim_cw = _cosine_similarity(vec_c, vec_w)
                delta_live = (sim_wu - sim_cu) if (sim_wu is not None and sim_cu is not None) else None
                info_lines = [
                    f"Method: {method_label}",
                    f"Pair: {match_idx + 1}/{limit_pairs}",
                    f"Frames C/U/W: {ref_idx}/{usr_idx}/{wrong_ref_idx if wrong_mode != 'none' else 'NA'}",
                    f"Wrong pick mode: {wrong_mode}",
                    f"C-U: {_fmt_score(sim_cu)}",
                    f"W-U: {_fmt_score(sim_wu)}",
                    f"Delta: {_fmt_score(delta_live)}",
                    f"C-W: {_fmt_score(sim_cw)}",
                    f"Hands C/U/W: {n_hands_correct}/{n_hands_user}/{n_hands_wrong}",
                ]
                if view == "m4":
                    m4_kind = str(match_data.get("m4_kind", "bones_angles_len_d1"))
                    comp = _m4_component_sims(vec_c, vec_u, vec_w, m4_kind)
                    info_lines.extend([
                        f"M4 kind: {m4_kind}",
                        f"M4 ang(C-U): {_fmt_score(comp.get('ang_cu'))}",
                        f"M4 ang(W-U): {_fmt_score(comp.get('ang_wu'))}",
                        f"M4 len(C-U): {_fmt_score(comp.get('len_cu'))}",
                        f"M4 len(W-U): {_fmt_score(comp.get('len_wu'))}",
                    ])

                latest_canvas = compose_view_four(
                    frame_correct,
                    frame_user,
                    frame_wrong,
                    row,
                    case_idx,
                    total_cases,
                    panel_w,
                    panel_h,
                    f"{correct_title}  F{ref_idx}  sim={sim_correct:.3f}",
                    f"{user_title}  F{usr_idx}",
                    (
                        f"{wrong_title}  ({wrong_mode})"
                        if wrong_mode == "none"
                        else (
                            f"{wrong_title}  ({wrong_mode})  F{wrong_ref_idx}  sim={sim_wrong:.3f}"
                            + (f"  d={delta:+.3f}" if delta is not None else "")
                        )
                    ),
                    n_hands_correct,
                    n_hands_user,
                    n_hands_wrong,
                    "Extra Info",
                    info_lines,
                    sim_cu,
                    sim_wu,
                )

                # Compact status line
                status = f"Mode=MATCH  Auto={'ON' if match_autoplay else 'OFF'}  |  H help  |  M time  N next  Q quit"
                _put_line(latest_canvas, status, 10, latest_canvas.shape[0] - 12, scale=0.48, color=UI_COLOR_MUTED, thickness=1)

                if help_visible:
                    help_lines = [
                        "Keys (Match): Space autoplay, A/D step, 4 M4V, 6 M6",
                        "Meaning: C-U=cosine(correct_frame_feat, user_frame_feat)",
                        "         W-U=cosine(wrong_frame_feat, user_frame_feat)",
                        "         Δ = W-U - C-U  (positive => wrong matches better)",
                        f"View: {view.upper()} | Wrong fallback: {wrong_fallback}",
                    ]
                    box_w = min(latest_canvas.shape[1] - 20, 980)
                    box_h = 18 * len(help_lines) + 18
                    _draw_box(latest_canvas, 10, HEADER_H + 40, box_w, box_h, UI_BG, alpha=0.78)
                    y0 = HEADER_H + 60
                    for line in help_lines:
                        _put_line(latest_canvas, line, 24, y0, scale=0.52, color=UI_COLOR_TEXT, thickness=1)
                        y0 += 18

                if writer is not None:
                    writer.write(latest_canvas)
                    frames_written += 1

                if display:
                    preview = _fit_canvas_for_display(latest_canvas)
                    cv2.imshow("Top-5 Miss Debug Flow (Correct | User | Wrong | Extra Info)", preview)
                    key = cv2.waitKey(wait_ms if match_autoplay else 0) & 0xFF
                    if key == ord("q"):
                        return "quit"
                    if key == ord("n"):
                        return "next"
                    if key == ord("m") or key == ord("t") or key == ord("T"):
                        mode = "time"
                    if key == ord(" "):
                        match_autoplay = not match_autoplay
                    if key == ord("h") or key == ord("H") or key == ord("?"):
                        help_visible = not help_visible
                    if key == ord("1"):
                        pass
                    if key == ord("2"):
                        pass
                    if key == ord("3"):
                        pass
                    if key == ord("4"):
                        method_view = "m4"
                    if key == ord("5"):
                        pass
                    if key == ord("6"):
                        method_view = "m6"
                    if key == ord("d") or key == 83:
                        match_idx = min(match_idx + 1, limit_pairs - 1)
                    if key == ord("a") or key == 81:
                        match_idx = max(match_idx - 1, 0)
                else:
                    # export mode: always iterate through all pairs
                    match_idx += 1
                    if match_idx >= limit_pairs:
                        break
    finally:
        cap_correct.release()
        cap_user.release()
        cap_wrong.release()
        hands_correct.close()
        hands_user.close()
        hands_wrong.close()
        if writer is not None:
            writer.release()

    if mode == "time":
        print(f"  Processed {frame_idx} frames")
    else:
        print(f"  Wrote {frames_written} frames (match mode)")
    return "next"


def pick_wrong_detected_folder(row: Dict) -> str:
    correct = str(row.get("folder", ""))
    top1 = str(row.get("top1_folder", ""))
    if normalize_key(top1) != normalize_key(correct):
        return top1
    for item in (row.get("top5_rankings", []) or []):
        if isinstance(item, (list, tuple)) and item:
            candidate = str(item[0])
            if normalize_key(candidate) != normalize_key(correct):
                return candidate
    return top1


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="4-panel MediaPipe debug flow for Top-5 misses."
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=base / "recognition_results" / "recognition_dtw_summary.json",
        help="Path to recognition summary JSON generated by evaluate_recognition.py",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=base / "Videos",
        help="Root Videos directory containing phrase folders.",
    )
    parser.add_argument("--start-index", type=int, default=1, help="1-based case index in the filtered miss list.")
    parser.add_argument("--max-cases", type=int, default=0, help="Limit number of miss cases to play (0 = all).")
    parser.add_argument("--panel-width", type=int, default=520, help="Width of each side panel.")
    parser.add_argument("--panel-height", type=int, default=480, help="Height of each side panel.")
    parser.add_argument("--fps", type=float, default=0.0, help="Playback fps override (0 = auto from videos).")
    parser.add_argument("--mode", choices=["time", "match"], default="time", help="Start mode: normal playback or DTW matched-frame view.")
    parser.add_argument("--method-view", choices=["m4", "m6"], default="m4", help="In match mode, initial method view.")
    parser.add_argument("--dtw-window", type=float, default=0.3, help="DTW window ratio (must match evaluation if you want identical behavior).")
    parser.add_argument("--subseq-window", type=float, default=0.25, help="Subsequence DTW window ratio (camera-like; default matches streaming_dtw).")
    parser.add_argument("--similarity-threshold", type=float, default=0.99, help="Redundant-frame drop threshold (must match evaluation).")
    parser.add_argument("--cache-dir", type=Path, default=base / "cache" / "debug_viewer", help="Disk cache for features/alignments.")
    parser.add_argument("--no-disk-cache", action="store_true", help="Disable disk caching (recompute every run).")
    parser.add_argument("--recompute", action="store_true", help="Ignore disk cache and recompute, but still write fresh cache.")
    parser.add_argument("--max-pairs", type=int, default=0, help="In match mode, limit to first N aligned pairs (0 = all).")
    parser.add_argument("--wrong-fallback", choices=["exact", "nearest"], default="exact",
                        help="In match mode, how to pick wrong(top1) ref frame for a given user frame.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory to export 4-panel mp4 files.")
    parser.add_argument("--no-display", action="store_true", help="Disable live window display.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths only, do not play videos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.results_json.exists():
        raise FileNotFoundError(f"Results JSON not found: {args.results_json}")
    if not args.videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {args.videos_dir}")

    results = load_results(args.results_json)
    detailed = results.get("detailed", [])
    misses = select_top5_misses(detailed)

    if not misses:
        print("No Top-5 miss cases found in results JSON.")
        return

    start_zero = max(0, args.start_index - 1)
    selected = misses[start_zero:]
    if args.max_cases > 0:
        selected = selected[: args.max_cases]

    exact_index, normalized_index = build_folder_index(args.videos_dir)

    total = len(selected)
    print(f"Top-5 misses available: {len(misses)} | Selected for playback: {total}")
    print("Panel order: Correct Translator | User | Wrong Top-1 Translator | Extra Info")

    display = not args.no_display
    if not display and args.save_dir is None and not args.dry_run:
        print("Display is disabled and save-dir is not set. Nothing to render.")
        return

    feature_cache: Dict[str, Tuple[np.ndarray, List[int]]] = {}
    disk_cache = DiskCache(
        root=args.cache_dir,
        enabled=(not args.no_disk_cache),
        recompute=bool(args.recompute),
    )

    for i, row in enumerate(selected, start=1):
        correct_folder = str(row.get("folder", ""))
        user_name = str(row.get("user_video", ""))
        wrong_folder = pick_wrong_detected_folder(row)

        user_folder_path = resolve_folder_path(correct_folder, exact_index, normalized_index)
        if user_folder_path is None:
            print(f"[{i}/{total}] [skip] Folder not found: {correct_folder}")
            continue

        correct_folder_path = resolve_folder_path(correct_folder, exact_index, normalized_index)
        if correct_folder_path is None:
            print(f"[{i}/{total}] [skip] Correct folder not found: {correct_folder}")
            continue

        wrong_folder_path = resolve_folder_path(wrong_folder, exact_index, normalized_index)
        if wrong_folder_path is None:
            print(f"[{i}/{total}] [skip] Wrong-top1 folder not found: {wrong_folder}")
            continue

        correct_video = find_translator_video(correct_folder_path)
        if correct_video is None:
            print(f"[{i}/{total}] [skip] Correct translator video not found in: {correct_folder_path}")
            continue

        wrong_video = find_translator_video(wrong_folder_path)
        if wrong_video is None:
            print(f"[{i}/{total}] [skip] Wrong-top1 translator video not found in: {wrong_folder_path}")
            continue

        user_video = find_user_video(user_folder_path, user_name)
        if user_video is None:
            print(f"[{i}/{total}] [skip] User video not found: {user_name}")
            continue

        correct_title = "Correct Translator"
        user_title = "User"
        wrong_title = "Wrong Detected Top1"

        print(
            f"[{i}/{total}] rank={row.get('correct_rank', '?')} | "
            f"correct={correct_video.name} | user={user_video.name} | wrong_top1={wrong_video.name}"
        )

        if args.dry_run:
            continue

        save_path = None
        if args.save_dir is not None:
            case_id = f"case_{i:03d}_rank_{int(row.get('correct_rank', 9999)):03d}"
            save_path = args.save_dir / f"{case_id}.mp4"

        status = play_case_three(
            row=row,
            case_idx=i,
            total_cases=total,
            correct_video=correct_video,
            user_video=user_video,
            wrong_video=wrong_video,
            correct_title=correct_title,
            user_title=user_title,
            wrong_title=wrong_title,
            panel_w=args.panel_width,
            panel_h=args.panel_height,
            fps_override=args.fps,
            display=display,
            save_path=save_path,
            mode=args.mode,
            method_view=args.method_view,
            dtw_window_ratio=args.dtw_window,
            subseq_window_ratio=args.subseq_window,
            similarity_threshold=args.similarity_threshold,
            feature_cache=feature_cache,
            disk_cache=disk_cache,
            max_pairs=args.max_pairs,
            wrong_fallback=args.wrong_fallback,
        )
        if status == "quit":
            break

    if display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
