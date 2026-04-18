"""
Real-time Sign Language Similarity Matching

Opens the webcam, auto-detects when you start signing (hands visible),
records until hands disappear for ~2 seconds, then finds the top 3
most similar pre-computed videos and displays them on screen + terminal.

Controls:
    q / ESC  – Quit
    r        – Reset (discard current recording, start fresh)

Usage:
    python realtime_similarity.py
"""

import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from inference.similarity.feature_extractor import HandFeatureExtractor

# ── Config ─────────────────────────────────────────────
MATRICES_DIR = Path("data/processed/matrices")
CAMERA_INDEX = 0
N_FRAMES = 64            # must match what was used for extraction
MAX_HANDS = 2
HAND_GONE_TIMEOUT = 2.0  # seconds without hands → stop recording
TOP_K = 3
# ───────────────────────────────────────────────────────

# ── State machine ──────────────────────────────────────
STATE_IDLE = "IDLE"           # Waiting for hands
STATE_RECORDING = "RECORDING" # Hands detected, recording frames
STATE_COOLDOWN = "COOLDOWN"   # Hands gone, waiting to confirm stop
STATE_MATCHING = "MATCHING"   # Computing similarity
# ───────────────────────────────────────────────────────


def load_all_vectors(matrices_dir: Path):
    """
    Load every .npy from matrices_dir.
    Returns dict: { sentence_label: list_of (npy_path, matrix) }
    """
    db = {}  # label -> [(path, matrix), ...]
    npy_files = sorted(matrices_dir.rglob("*.npy"))
    for p in npy_files:
        rel = p.relative_to(matrices_dir)
        label = rel.parts[0] if len(rel.parts) > 1 else rel.stem
        matrix = np.load(p)
        db.setdefault(label, []).append((str(rel), matrix))
    return db


def compute_cosine_similarity(query: np.ndarray, ref: np.ndarray) -> float:
    """Cosine similarity on flattened feature matrices."""
    v1 = query.flatten()
    v2 = ref.flatten()
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(1.0 - cosine(v1, v2))


def find_top_k(query_matrix: np.ndarray, db: dict, k: int = 3):
    """
    Compare query against all stored vectors using cosine similarity.
    Returns list of (label, best_score, best_file) sorted by score desc.
    """
    label_scores = []
    for label, entries in db.items():
        best_score = -1.0
        best_file = ""
        for rel_path, ref_matrix in entries:
            score = compute_cosine_similarity(query_matrix, ref_matrix)
            if score > best_score:
                best_score = score
                best_file = rel_path
        label_scores.append((label, best_score, best_file))

    label_scores.sort(key=lambda x: x[1], reverse=True)
    return label_scores[:k]


def sample_frames(frames: list, target: int) -> list:
    """Uniform-sample or duplicate frames to exactly `target` count."""
    n = len(frames)
    if n == 0:
        return [np.zeros((480, 640, 3), dtype=np.uint8)] * target
    if n == target:
        return frames
    indices = [int(i * n / target) for i in range(target)]
    indices = [min(idx, n - 1) for idx in indices]
    return [frames[idx] for idx in indices]


def trim_no_hand_edges(frames: list, detector) -> list:
    """
    Trim frames from the start and end that have no hands.
    Keeps only the contiguous core where hands are present.
    Falls back to all frames if nothing remains.
    """
    has_hand = []
    for f in frames:
        res = detector.process(f)
        has_hand.append(res.multi_hand_landmarks is not None)

    # Find first and last frame with hands
    first = next((i for i, h in enumerate(has_hand) if h), 0)
    last = next((i for i, h in reversed(list(enumerate(has_hand))) if h), len(frames) - 1)

    trimmed = frames[first:last + 1]
    return trimmed if len(trimmed) > 0 else frames


def put_multiline_text(img, lines, org, font, scale, color, thickness, line_gap=30):
    """Draw multiple lines of text on an image."""
    x, y = org
    for line in lines:
        cv2.putText(img, line, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        y += line_gap


def main():
    # ── Load pre-computed vectors ──────────────────────
    print("Loading pre-computed vectors...", flush=True)
    db = load_all_vectors(MATRICES_DIR)
    total_vecs = sum(len(v) for v in db.values())
    print(f"Loaded {total_vecs} vectors across {len(db)} sentences.", flush=True)

    # ── Init MediaPipe for hand detection ──────────────
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils

    # ── Init feature extractor ─────────────────────────
    feat_ext = HandFeatureExtractor(max_hands=MAX_HANDS, normalize=True)

    # ── Open camera ────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera", flush=True)
        sys.exit(1)

    print("Camera opened. Show your hands to start signing!", flush=True)
    print("Press 'q' or ESC to quit, 'r' to reset.\n", flush=True)

    # ── State ──────────────────────────────────────────
    state = STATE_IDLE
    recorded_frames = []        # RGB frames collected while signing
    last_hand_time = 0.0        # timestamp of last frame with hand
    top_results = []            # latest top-K results
    match_count = 0             # how many matches we've done

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = hands_detector.process(frame_rgb)
        hands_visible = results.multi_hand_landmarks is not None
        now = time.time()

        # Draw hand landmarks on display frame
        display = frame_bgr.copy()
        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(display, hand_lm, mp_hands.HAND_CONNECTIONS)

        # ── State machine ─────────────────────────────
        if state == STATE_IDLE:
            if hands_visible:
                state = STATE_RECORDING
                recorded_frames = [frame_rgb]
                last_hand_time = now
                print(f"[Recording] Hands detected – recording started...", flush=True)

        elif state == STATE_RECORDING:
            recorded_frames.append(frame_rgb)
            if hands_visible:
                last_hand_time = now
            else:
                # Hands gone, move to cooldown
                state = STATE_COOLDOWN

        elif state == STATE_COOLDOWN:
            recorded_frames.append(frame_rgb)
            if hands_visible:
                last_hand_time = now
                state = STATE_RECORDING  # hands came back
            elif (now - last_hand_time) >= HAND_GONE_TIMEOUT:
                # Hands gone long enough → process
                state = STATE_MATCHING

        if state == STATE_MATCHING:
            match_count += 1
            n_recorded = len(recorded_frames)
            print(f"\n[Matching #{match_count}] Recorded {n_recorded} frames", flush=True)

            # Trim leading/trailing frames without hands for cleaner signal
            trimmed = trim_no_hand_edges(recorded_frames, hands_detector)
            print(f"  After trimming: {len(trimmed)} hand-frames (was {n_recorded})", flush=True)

            # Sample to N_FRAMES and extract features
            sampled = sample_frames(trimmed, N_FRAMES)
            query_matrix = feat_ext.frames_to_feature_matrix(sampled)

            # Find top K
            top_results = find_top_k(query_matrix, db, TOP_K)

            # Print results
            print(f"  Top {TOP_K} matches:", flush=True)
            for rank, (label, score, fpath) in enumerate(top_results, 1):
                print(f"    #{rank}  {score:.4f}  {label}", flush=True)

            # Reset for next sign
            recorded_frames = []
            state = STATE_IDLE
            print(f"[Idle] Ready for next sign...\n", flush=True)

        # ── Draw overlay ──────────────────────────────
        # Status bar
        status_color = {
            STATE_IDLE: (200, 200, 200),
            STATE_RECORDING: (0, 0, 255),
            STATE_COOLDOWN: (0, 165, 255),
            STATE_MATCHING: (255, 255, 0),
        }[state]
        cv2.rectangle(display, (0, 0), (display.shape[1], 36), (40, 40, 40), -1)
        status_text = state
        if state == STATE_RECORDING:
            status_text = f"RECORDING  ({len(recorded_frames)} frames)"
        elif state == STATE_COOLDOWN:
            remaining = HAND_GONE_TIMEOUT - (now - last_hand_time)
            status_text = f"COOLDOWN  ({remaining:.1f}s)"
        cv2.putText(display, status_text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                     0.7, status_color, 2, cv2.LINE_AA)

        # Top-K results overlay
        if top_results:
            h = display.shape[0]
            y_start = h - 30 - TOP_K * 40
            cv2.rectangle(display, (0, y_start - 10), (display.shape[1], h), (40, 40, 40), -1)
            cv2.putText(display, f"Top {TOP_K} Matches:", (10, y_start + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            for rank, (label, score, _) in enumerate(top_results, 1):
                y = y_start + 15 + rank * 35
                # Color by rank: green → yellow → orange
                colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255)]
                color = colors[rank - 1] if rank <= len(colors) else (200, 200, 200)
                text = f"#{rank}  {score:.3f}  {label}"
                # Truncate long labels
                if len(text) > 70:
                    text = text[:67] + "..."
                cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 2, cv2.LINE_AA)

        cv2.imshow("Sign Language - Real-time Similarity", display)

        # ── Key handling ──────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord("r"):
            recorded_frames = []
            top_results = []
            state = STATE_IDLE
            print("[Reset] Cleared. Ready for next sign.", flush=True)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    feat_ext.__del__()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
