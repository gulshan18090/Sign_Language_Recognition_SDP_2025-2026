"""
Real-Time Camera-Based Sign Language Similarity using Streaming DTW

Workflow:
  1. Pre-compute reference features for all 3 translator videos
  2. Open camera, show "Waiting for hand..."
  3. When hand detected → start recording & streaming DTW against all references
  4. Overlay live running similarity scores on camera feed
  5. When hand disappears for 4 seconds → stop, finalize alignment
  6. Show ranked results, ask user to approve or retry

Controls:
  [R] Retry (reset and record again)
  [Q] Quit
  [A] Approve result (saves summary)
"""

import numpy as np
import cv2
import os
import sys
import time
import glob
import json
import pickle
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.dirname(__file__))
from similarity.feature_extractor import HandFeatureExtractor
from streaming_dtw import (
    StreamingDTW, precompute_reference, drop_similar_frames,
    compute_cost_matrix, subsequence_dtw_accumulated, subsequence_dtw_backtrack
)


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEOS_DIR = os.path.join(BASE_DIR, "Videos")
MATRICES_DIR = os.path.join(BASE_DIR, "matrices")
CACHE_DIR = os.path.join(BASE_DIR, "cache")

HAND_GONE_TIMEOUT = 4.0       # seconds without hand → stop recording
REDUNDANCY_THRESHOLD = 0.99   # for dropping similar frames
MIN_FRAMES_FOR_DTW = 5        # need at least this many kept frames

# Auto-detect folders with translator videos
def get_all_reference_folders():
    """Auto-detect all folders in Videos/ that contain translator videos."""
    folders = []
    if not os.path.exists(VIDEOS_DIR):
        return folders
    
    for folder_name in os.listdir(VIDEOS_DIR):
        folder_path = os.path.join(VIDEOS_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        
        # Check if folder has a translator video
        for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
            matches = glob.glob(os.path.join(folder_path, ext))
            for m in matches:
                if "translator" in os.path.basename(m).lower():
                    folders.append(folder_name)
                    break
            if folder_name in folders:
                break
    
    return sorted(folders)

FOLDERS = get_all_reference_folders()

# Ensure cache directory exists
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


# =============================================================================
# Reference video discovery
# =============================================================================

def find_translator_video(folder):
    """Find the translator (reference) video in a folder."""
    folder_path = os.path.join(VIDEOS_DIR, folder)
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        matches = glob.glob(os.path.join(folder_path, ext))
        for m in matches:
            if "translator" in os.path.basename(m).lower():
                return m
    # Fallback: first video file
    for ext in ["*.mp4", "*.avi", "*.mov", "*.mkv"]:
        matches = glob.glob(os.path.join(folder_path, ext))
        if matches:
            return matches[0]
    return None


def get_cache_path(folder, video_path):
    """Get cache file path for a reference video."""
    video_name = os.path.basename(video_path)
    cache_name = f"ref_{folder}_{video_name}.pkl"
    return os.path.join(CACHE_DIR, cache_name)


def save_reference_cache(folder, video_path, features, indices):
    """Save pre-computed reference features to cache."""
    cache_path = get_cache_path(folder, video_path)
    cache_data = {
        'features': features,
        'indices': indices,
        'video_path': video_path,
        'video_name': os.path.basename(video_path),
        'folder': folder,
        'timestamp': time.time()
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    print(f"  Cached to: {cache_path}")


def load_reference_cache(folder, video_path):
    """Load pre-computed reference features from cache.
    
    Returns:
        dict or None: cached data if exists and valid, None otherwise
    """
    cache_path = get_cache_path(folder, video_path)
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Verify cache is for the same video
        if cache_data['video_path'] == video_path:
            return cache_data
        else:
            print(f"  Cache invalid: video path mismatch")
            return None
    except Exception as e:
        print(f"  Cache load error: {e}")
        return None


def precompute_all_references():
    """Pre-compute reference features for all translator videos.
    Uses cached features if available, otherwise computes and caches.

    Returns:
        dict: folder → {
            'features': np.ndarray (N, 126),
            'indices': list,
            'video_path': str,
            'video_name': str
        }
    """
    references = {}
    for folder in FOLDERS:
        video_path = find_translator_video(folder)
        if video_path is None:
            print(f"  WARNING: No translator video found in {folder}")
            continue

        print(f"\n--- Folder {folder}: {os.path.basename(video_path)} ---")
        
        # Try to load from cache first
        cached = load_reference_cache(folder, video_path)
        if cached is not None:
            print(f"  Loaded from cache: {len(cached['features'])} reference frames")
            references[folder] = {
                'features': cached['features'],
                'indices': cached['indices'],
                'video_path': cached['video_path'],
                'video_name': cached['video_name'],
            }
            continue
        
        # Cache miss - compute features
        print(f"  Computing features (no cache found)...")
        features, indices = precompute_reference(video_path)
        if len(features) == 0:
            print(f"  WARNING: No features extracted for {folder}")
            continue

        references[folder] = {
            'features': features,
            'indices': indices,
            'video_path': video_path,
            'video_name': os.path.basename(video_path),
        }
        print(f"  Computed: {len(features)} reference frames")
        
        # Save to cache for next time
        save_reference_cache(folder, video_path, features, indices)

    return references


# =============================================================================
# PIL-based Unicode text rendering
# =============================================================================

def _get_font(size=18):
    """Get a font that supports Azerbaijani characters."""
    font_paths = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/tahoma.ttf",
        "C:/Windows/Fonts/calibri.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


def put_text_unicode(frame, text, pos, font_size=18, color=(255, 255, 255)):
    """Draw Unicode text on an OpenCV frame using PIL."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    draw.text(pos, text, font=font, fill=color)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(frame, result)


# =============================================================================
# UI Drawing helpers
# =============================================================================

def draw_text_with_bg(frame, text, pos, font_scale=0.7, color=(255, 255, 255),
                      bg_color=(0, 0, 0), thickness=2, padding=8):
    """Draw text with a background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x - padding, y - th - padding),
                  (x + tw + padding, y + baseline + padding), bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)


def draw_status_bar(frame, text, color=(50, 50, 50)):
    """Draw a full-width status bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 50
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (15, h - 15), font, 0.6,
                (255, 255, 255), 2)


def draw_scores_panel(frame, scores, state):
    """Draw live scores panel on the right side of the frame."""
    h, w = frame.shape[:2]
    panel_w = 280
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, min(h, 50 + len(scores) * 60)),
                  (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30
    cv2.putText(frame, "Live Scores:", (w - panel_w + 10, y_offset),
                font, 0.6, (200, 200, 200), 1)
    y_offset += 15

    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for i, (folder, score) in enumerate(sorted_scores):
        y_offset += 45
        # Color based on score
        if score >= 0.7:
            bar_color = (0, 200, 0)    # Green
        elif score >= 0.45:
            bar_color = (0, 200, 200)  # Yellow
        else:
            bar_color = (0, 0, 200)    # Red

        # Score bar
        bar_x = w - panel_w + 10
        bar_w = int((panel_w - 30) * min(score, 1.0))
        cv2.rectangle(frame, (bar_x, y_offset - 12),
                      (bar_x + bar_w, y_offset + 8), bar_color, -1)
        cv2.rectangle(frame, (bar_x, y_offset - 12),
                      (bar_x + panel_w - 30, y_offset + 8), (100, 100, 100), 1)

        label = f"Ref {folder}: {score:.3f}"
        cv2.putText(frame, label, (bar_x + 5, y_offset + 5),
                    font, 0.5, (255, 255, 255), 1)


def draw_recording_indicator(frame, elapsed):
    """Draw a red recording dot and elapsed time."""
    h, w = frame.shape[:2]
    # Blinking red circle
    if int(elapsed * 2) % 2 == 0:
        cv2.circle(frame, (30, 30), 12, (0, 0, 255), -1)
    cv2.putText(frame, f"REC {elapsed:.1f}s", (50, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def draw_hand_lost_countdown(frame, seconds_left):
    """Show countdown when hand is lost."""
    h, w = frame.shape[:2]
    text = f"Hand lost... stopping in {seconds_left:.1f}s"
    draw_text_with_bg(frame, text, (w // 2 - 200, h // 2),
                      font_scale=0.8, color=(0, 100, 255),
                      bg_color=(0, 0, 0))


def draw_results_screen(frame, results, references, scroll_offset=0):
    """Draw clean, beautiful results screen with proper Unicode support."""
    h, w = frame.shape[:2]

    # Dark background
    frame[:] = (30, 30, 35)

    # Title bar
    cv2.rectangle(frame, (0, 0), (w, 60), (45, 45, 50), -1)
    put_text_unicode(frame, "RESULTS", (w // 2 - 55, 15), font_size=32, color=(255, 255, 255))
    cv2.line(frame, (20, 62), (w - 20, 62), (80, 80, 90), 2)

    # Sort by score
    sorted_results = sorted(results.items(),
                            key=lambda x: x[1]['score'], reverse=True)

    # Show only top 10 with scrolling
    visible_start = scroll_offset
    visible_end = min(len(sorted_results), scroll_offset + 8)
    visible = sorted_results[visible_start:visible_end]

    y = 80
    medal_colors = [(0, 215, 255), (192, 192, 192), (80, 130, 210)]  # Gold, Silver, Bronze

    for display_idx, (folder, res) in enumerate(visible):
        rank = visible_start + display_idx + 1
        score = res['score']
        n_aligned = res.get('n_aligned', 0)
        n_ref = res.get('n_ref', 0)

        # Row background
        row_color = (40, 40, 48) if rank % 2 == 0 else (35, 35, 42)
        cv2.rectangle(frame, (10, y), (w - 10, y + 50), row_color, -1)

        # Medal / rank
        if rank <= 3:
            mc = medal_colors[rank - 1]
            cv2.circle(frame, (35, y + 25), 14, mc, -1)
            cv2.putText(frame, str(rank), (30, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            cv2.putText(frame, f"#{rank}", (22, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (130, 130, 140), 1)

        # Score bar
        bar_x, bar_y = 60, y + 8
        bar_w = 120
        bar_h = 20
        if score >= 0.7:
            bar_color = (0, 190, 0)
        elif score >= 0.45:
            bar_color = (0, 190, 190)
        else:
            bar_color = (60, 60, 190)

        filled_w = int(bar_w * min(score, 1.0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 70), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (90, 90, 100), 1)

        # Score text
        cv2.putText(frame, f"{score:.4f}", (bar_x + 5, bar_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # Folder name (Unicode via PIL)
        folder_display = folder[:40] + "..." if len(folder) > 40 else folder
        put_text_unicode(frame, folder_display, (195, y + 6), font_size=15, color=(230, 230, 240))

        # Aligned frames info
        cv2.putText(frame, f"{n_aligned}/{n_ref} frames",
                    (195, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 150), 1)

        # Top 3 get [V] view button marker
        if rank <= 3:
            btn_x = w - 70
            cv2.rectangle(frame, (btn_x, y + 10), (btn_x + 50, y + 40), (80, 130, 60), -1)
            cv2.rectangle(frame, (btn_x, y + 10), (btn_x + 50, y + 40), (120, 180, 80), 1)
            cv2.putText(frame, f"[{rank}]", (btn_x + 10, y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        y += 55

    # Scroll indicators
    if scroll_offset > 0:
        cv2.putText(frame, "^ Scroll Up [W]", (w // 2 - 60, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 160), 1)
    if visible_end < len(sorted_results):
        cv2.putText(frame, f"v Scroll Down [S] ({len(sorted_results) - visible_end} more)",
                    (w // 2 - 80, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 160), 1)

    # Bottom bar
    cv2.rectangle(frame, (0, h - 45), (w, h), (45, 45, 50), -1)
    cv2.line(frame, (20, h - 45), (w - 20, h - 45), (80, 80, 90), 2)
    controls = "[1][2][3] Compare Frames  |  [W][S] Scroll  |  [A] Approve  [R] Retry  [Q] Quit"
    cv2.putText(frame, controls, (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 210), 1)


# =============================================================================
# Frame-by-frame comparison window
# =============================================================================

def extract_frame_at_index(video_path, frame_idx):
    """Extract a single frame from video at a given index."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return frame
    return None


def show_frame_comparison(folder, results, references, user_frames_bgr):
    """Open a separate window showing frame-by-frame DTW alignment for a match.
    
    Args:
        folder: folder name of the match
        results: dict of all results
        references: dict of all references
        user_frames_bgr: list of user's recorded camera frames (BGR)
    """
    if folder not in results or folder not in references:
        print(f"  No data for {folder}")
        return

    res = results[folder]
    ref = references[folder]
    path = res.get('path', [])
    frame_sims = res.get('frame_sims', [])

    if not path:
        print(f"  No alignment path for {folder}")
        return

    ref_video = ref['video_path']
    n_pairs = len(path)
    current_pair = 0

    win_name = f"Frame Comparison: {folder[:50]}"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 800, 500)

    while True:
        ref_orig_idx, user_orig_idx, sim = path[current_pair]

        # Get reference frame from video
        ref_frame = extract_frame_at_index(ref_video, ref_orig_idx)
        if ref_frame is None:
            ref_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(ref_frame, "No Frame", (80, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Get user frame from stored frames
        if user_orig_idx < len(user_frames_bgr):
            user_frame = user_frames_bgr[user_orig_idx]
        else:
            user_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            cv2.putText(user_frame, "No Frame", (80, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Resize both to same height
        target_h = 360
        ref_frame = cv2.resize(ref_frame, (int(ref_frame.shape[1] * target_h / ref_frame.shape[0]), target_h))
        user_frame = cv2.resize(user_frame, (int(user_frame.shape[1] * target_h / user_frame.shape[0]), target_h))

        # Similarity color
        if sim >= 0.7:
            sim_color = (0, 200, 0)
        elif sim >= 0.45:
            sim_color = (0, 200, 200)
        else:
            sim_color = (0, 0, 200)

        # Border color
        border = 4
        ref_bordered = cv2.copyMakeBorder(ref_frame, border, border, border, border,
                                          cv2.BORDER_CONSTANT, value=sim_color)
        user_bordered = cv2.copyMakeBorder(user_frame, border, border, border, border,
                                           cv2.BORDER_CONSTANT, value=sim_color)

        # Separator
        sep = np.full((ref_bordered.shape[0], 8, 3), (50, 50, 55), dtype=np.uint8)

        # Combine side by side
        combined = np.hstack([ref_bordered, sep, user_bordered])

        # Add header
        header_h = 60
        header = np.full((header_h, combined.shape[1], 3), (35, 35, 40), dtype=np.uint8)

        # Title
        put_text_unicode(header, f"{folder[:55]}", (10, 5), font_size=16, color=(200, 200, 210))

        # Pair info
        pair_text = f"Pair {current_pair + 1}/{n_pairs}  |  Ref F{ref_orig_idx} <-> User F{user_orig_idx}  |  Sim: {sim:.4f}"
        cv2.putText(header, pair_text, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, sim_color, 1)

        # Labels
        ref_label_x = combined.shape[1] // 4 - 30
        user_label_x = combined.shape[1] * 3 // 4 - 30
        cv2.putText(header, "REFERENCE", (ref_label_x, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 190), 1)
        cv2.putText(header, "YOUR SIGN", (user_label_x, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 190), 1)

        # Footer with controls
        footer_h = 35
        footer = np.full((footer_h, combined.shape[1], 3), (35, 35, 40), dtype=np.uint8)
        cv2.putText(footer, "[A/D] Prev/Next  |  [Space] Play All  |  [ESC] Close",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 170), 1)

        # Score bar at bottom of footer
        bar_x = combined.shape[1] - 160
        bar_w = 140
        filled = int(bar_w * min(sim, 1.0))
        cv2.rectangle(footer, (bar_x, 8), (bar_x + bar_w, 26), (60, 60, 70), -1)
        cv2.rectangle(footer, (bar_x, 8), (bar_x + filled, 26), sim_color, -1)

        display = np.vstack([header, combined, footer])
        cv2.imshow(win_name, display)

        key = cv2.waitKey(0) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('d') or key == 83:  # D or Right arrow
            current_pair = min(current_pair + 1, n_pairs - 1)
        elif key == ord('a') or key == 81:  # A or Left arrow
            current_pair = max(current_pair - 1, 0)
        elif key == ord(' '):  # Space - auto play
            for p in range(n_pairs):
                current_pair = p
                # Rebuild display for this pair (simplified)
                r_idx, u_idx, s = path[p]
                rf = extract_frame_at_index(ref_video, r_idx)
                if rf is None:
                    rf = np.zeros((240, 320, 3), dtype=np.uint8)
                uf = user_frames_bgr[u_idx] if u_idx < len(user_frames_bgr) else np.zeros((240, 320, 3), dtype=np.uint8)
                rf = cv2.resize(rf, (int(rf.shape[1] * target_h / rf.shape[0]), target_h))
                uf = cv2.resize(uf, (int(uf.shape[1] * target_h / uf.shape[0]), target_h))
                sc = (0, 200, 0) if s >= 0.7 else (0, 200, 200) if s >= 0.45 else (0, 0, 200)
                rfb = cv2.copyMakeBorder(rf, border, border, border, border, cv2.BORDER_CONSTANT, value=sc)
                ufb = cv2.copyMakeBorder(uf, border, border, border, border, cv2.BORDER_CONSTANT, value=sc)
                sep2 = np.full((rfb.shape[0], 8, 3), (50, 50, 55), dtype=np.uint8)
                comb = np.hstack([rfb, sep2, ufb])
                hdr = np.full((header_h, comb.shape[1], 3), (35, 35, 40), dtype=np.uint8)
                put_text_unicode(hdr, f"{folder[:55]}", (10, 5), font_size=16, color=(200, 200, 210))
                pt = f"Pair {p + 1}/{n_pairs}  |  Ref F{r_idx} <-> User F{u_idx}  |  Sim: {s:.4f}"
                cv2.putText(hdr, pt, (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.5, sc, 1)
                ftr = np.full((footer_h, comb.shape[1], 3), (35, 35, 40), dtype=np.uint8)
                cv2.putText(ftr, "Playing... [ESC] Stop", (10, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 170), 1)
                disp = np.vstack([hdr, comb, ftr])
                cv2.imshow(win_name, disp)
                k = cv2.waitKey(150) & 0xFF
                if k == 27:
                    break

    cv2.destroyWindow(win_name)


# =============================================================================
# Main camera loop
# =============================================================================

class CameraDTWSession:
    """Manages a single camera recording + DTW session."""

    def __init__(self, references, extractor):
        self.references = references
        self.extractor = extractor
        self.scroll_offset = 0
        self.user_frames_bgr = []  # Store user camera frames for comparison
        self.top3_folders = []     # Top 3 folder names after finalization
        self.reset()

    def reset(self):
        """Reset for a new recording."""
        self.streamers = {}
        for folder, ref in self.references.items():
            self.streamers[folder] = StreamingDTW(
                ref['features'], ref['indices'],
                redundancy_threshold=REDUNDANCY_THRESHOLD
            )

        self.state = 'waiting'  # waiting | recording | hand_lost | results
        self.frame_idx = 0
        self.recording_start = None
        self.hand_last_seen = None
        self.results = None
        self.live_scores = {f: 0.0 for f in self.references}
        self.frames_kept = 0
        self.scroll_offset = 0
        self.user_frames_bgr = []
        self.top3_folders = []

    def process_camera_frame(self, frame_bgr):
        """Process one camera frame through all streaming DTW instances.

        Returns:
            display_frame: annotated BGR frame for display
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        display = frame_bgr.copy()
        now = time.time()

        if self.state == 'waiting':
            return self._state_waiting(display, frame_rgb, now)
        elif self.state == 'recording':
            return self._state_recording(display, frame_rgb, now)
        elif self.state == 'hand_lost':
            return self._state_hand_lost(display, frame_rgb, now)
        elif self.state == 'results':
            return self._state_results(display)

        return display

    def _state_waiting(self, display, frame_rgb, now):
        """Waiting for hand to appear."""
        feat = self.extractor.extract_features_from_frame(frame_rgb)
        has_hand = np.any(feat != 0)

        h, w = display.shape[:2]
        draw_text_with_bg(display, "Show your hand to start signing...",
                          (w // 2 - 230, h // 2 - 20),
                          font_scale=0.9, color=(0, 255, 0),
                          bg_color=(0, 0, 0))
        draw_status_bar(display, "Waiting for hand... | [Q] Quit")

        if has_hand:
            self.state = 'recording'
            self.recording_start = now
            self.hand_last_seen = now
            self.frame_idx = 0
            # Feed this first frame (features already extracted)
            self.frame_idx += 1
            for folder, streamer in self.streamers.items():
                result = streamer.process_feature_vector(feat, self.frame_idx)
                if result['kept']:
                    self.live_scores[folder] = result['running_score']
            any_streamer = next(iter(self.streamers.values()))
            self.frames_kept = any_streamer.n_kept

        return display

    def _state_recording(self, display, frame_rgb, now):
        """Recording: hand is visible, feeding frames to DTW."""
        has_hand = self._feed_frame(frame_rgb, now)

        elapsed = now - self.recording_start
        draw_recording_indicator(display, elapsed)
        draw_scores_panel(display, self.live_scores, 'recording')

        info = f"Frames: {self.frame_idx} | Kept: {self.frames_kept}"
        draw_status_bar(display, f"{info} | [Q] Quit")

        if not has_hand:
            self.state = 'hand_lost'
        else:
            self.hand_last_seen = now

        return display

    def _state_hand_lost(self, display, frame_rgb, now):
        """Hand disappeared — countdown to stop."""
        has_hand = self._feed_frame(frame_rgb, now)

        if has_hand:
            self.hand_last_seen = now
            self.state = 'recording'
            elapsed = now - self.recording_start
            draw_recording_indicator(display, elapsed)
            draw_scores_panel(display, self.live_scores, 'recording')
            draw_status_bar(display, f"Frames: {self.frame_idx} | [Q] Quit")
            return display

        time_since_hand = now - self.hand_last_seen
        seconds_left = HAND_GONE_TIMEOUT - time_since_hand

        elapsed = now - self.recording_start
        draw_recording_indicator(display, elapsed)
        draw_scores_panel(display, self.live_scores, 'hand_lost')
        draw_hand_lost_countdown(display, max(0, seconds_left))
        draw_status_bar(display, f"Frames: {self.frame_idx} | Hand lost...")

        if time_since_hand >= HAND_GONE_TIMEOUT:
            self._finalize()

        return display

    def _state_results(self, display):
        """Show final results."""
        draw_results_screen(display, self.results, self.references, self.scroll_offset)
        return display

    def scroll_results(self, direction):
        """Scroll results up/down."""
        if self.results is None:
            return
        max_scroll = max(0, len(self.results) - 8)
        self.scroll_offset = max(0, min(self.scroll_offset + direction, max_scroll))

    def show_comparison(self, rank_num):
        """Show frame comparison for top N match (1, 2, or 3)."""
        if rank_num < 1 or rank_num > 3:
            return
        if rank_num > len(self.top3_folders):
            print(f"  Only {len(self.top3_folders)} results available")
            return
        folder = self.top3_folders[rank_num - 1]
        print(f"\n  Opening frame comparison for #{rank_num}: {folder}")
        show_frame_comparison(folder, self.results, self.references, self.user_frames_bgr)

    def _feed_frame(self, frame_rgb, now):
        """Feed a frame to all StreamingDTW instances.

        Extracts features ONCE, then feeds the vector to all streamers.

        Returns:
            bool: True if hand was detected in this frame
        """
        self.frame_idx += 1

        # Extract features once (MediaPipe is the bottleneck)
        feat = self.extractor.extract_features_from_frame(frame_rgb)
        hand_detected = np.any(feat != 0)

        # Store raw camera frame for later comparison
        self.user_frames_bgr.append(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        for folder, streamer in self.streamers.items():
            result = streamer.process_feature_vector(feat, self.frame_idx)
            if result['kept']:
                self.live_scores[folder] = result['running_score']

        # Update kept count from any streamer (they all see the same frame)
        any_streamer = next(iter(self.streamers.values()))
        self.frames_kept = any_streamer.n_kept

        if hand_detected:
            self.hand_last_seen = now

        return hand_detected

    def _finalize(self):
        """Finalize all DTW instances and compute final scores."""
        print("\n" + "=" * 60)
        print("FINALIZING STREAMING DTW")
        print("=" * 60)

        self.results = {}
        for folder, streamer in self.streamers.items():
            if streamer.n_kept < MIN_FRAMES_FOR_DTW:
                print(f"\n  Folder {folder}: Too few frames ({streamer.n_kept})")
                self.results[folder] = {
                    'score': 0.0, 'n_ref': streamer.N,
                    'n_user_kept': streamer.n_kept, 'n_aligned': 0,
                    'message': 'Not enough frames'
                }
                continue

            print(f"\n--- Folder {folder} ---")
            print(f"  Reference: {self.references[folder]['video_name']}")
            print(f"  User frames received: {streamer.n_received}, "
                  f"kept: {streamer.n_kept}")

            res = streamer.finalize()
            self.results[folder] = res
            print(f"  SCORE: {res['score']:.4f}")

        # Print summary
        print("\n" + "-" * 60)
        print("RANKING:")
        sorted_res = sorted(self.results.items(),
                            key=lambda x: x[1]['score'], reverse=True)
        self.top3_folders = [f for f, _ in sorted_res[:3]]
        for rank, (folder, res) in enumerate(sorted_res, 1):
            print(f"  #{rank}  Folder {folder}: {res['score']:.4f}")
        print("-" * 60)

        self.state = 'results'

    def save_results(self):
        """Save results to JSON."""
        if self.results is None:
            return
        out = {}
        for folder, res in self.results.items():
            out[folder] = {
                'score': float(res['score']),
                'n_ref': int(res.get('n_ref', 0)),
                'n_user_kept': int(res.get('n_user_kept', 0)),
                'n_aligned': int(res.get('n_aligned', 0)),
                'ref_video': self.references[folder]['video_name'],
            }
        path = os.path.join(MATRICES_DIR, "camera_dtw_results.json")
        with open(path, 'w') as f:
            json.dump(out, f, indent=2)
        print(f"\nResults saved to {path}")


def main():
    print("=" * 60)
    print("REAL-TIME CAMERA SIGN LANGUAGE SIMILARITY")
    print("Using Streaming DTW with Pre-computed References")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Pre-compute all references (one-time cost)
    # ------------------------------------------------------------------
    print("\n[1/2] Pre-computing reference features...")
    references = precompute_all_references()
    if not references:
        print("ERROR: No reference videos found!")
        return

    print(f"\nReady: {len(references)} reference(s) loaded")
    for f, r in references.items():
        print(f"  Folder {f}: {len(r['features'])} frames — {r['video_name']}")

    # ------------------------------------------------------------------
    # Step 2: Open camera
    # ------------------------------------------------------------------
    print("\n[2/2] Opening camera...")
    extractor = HandFeatureExtractor(max_hands=2)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    session = CameraDTWSession(references, extractor)

    print("\nCamera ready. Show your hand to start!")
    print("Controls: [R] Retry  [A] Approve  [Q] Quit\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed")
                break

            # Mirror for more intuitive interaction
            frame = cv2.flip(frame, 1)

            display = session.process_camera_frame(frame)
            cv2.imshow("Sign Language Similarity - Streaming DTW", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                print("\nRetrying — reset recording...")
                session.reset()
            elif key == ord('a') and session.state == 'results':
                print("\nApproved!")
                session.save_results()
                print("Press [R] to record again or [Q] to quit.")
            elif key == ord('w') and session.state == 'results':
                session.scroll_results(-1)
            elif key == ord('s') and session.state == 'results':
                session.scroll_results(1)
            elif key == ord('1') and session.state == 'results':
                session.show_comparison(1)
            elif key == ord('2') and session.state == 'results':
                session.show_comparison(2)
            elif key == ord('3') and session.state == 'results':
                session.show_comparison(3)

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released. Done.")


if __name__ == "__main__":
    main()
