"""
Compute real dataset statistics + latency benchmark + translator/user accuracy split
for the paper's Section 4 (Experiments & Results).

Outputs all numbers to paper_figures/section4_stats.txt
"""

import sys, os, time, platform
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

VIDEOS_DIR = Path("data/raw/Videos")
MATRICES_DIR = Path("data/processed/matrices")
OUTPUT = Path("outputs/figures/paper_figures") / "section4_stats.txt"
OUTPUT.parent.mkdir(exist_ok=True)

lines = []  # collect output lines
def log(msg=""):
    print(msg, flush=True)
    lines.append(msg)


# ═══════════════════════════════════════════════════════════
# 1. DATASET STATISTICS
# ═══════════════════════════════════════════════════════════
log("=" * 70)
log("1. DATASET STATISTICS")
log("=" * 70)

total_videos = 0
total_frames = 0
total_duration = 0.0
sentence_counts = {}
all_durations = []
all_frame_counts = []
translator_count = 0
user_count = 0
unknown_count = 0

for sentence_folder in sorted(VIDEOS_DIR.iterdir()):
    if not sentence_folder.is_dir():
        continue
    label = sentence_folder.name
    vids_in_sentence = 0
    for vf in sorted(sentence_folder.iterdir()):
        if vf.suffix.lower() not in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
            continue
        cap = cv2.VideoCapture(str(vf))
        if not cap.isOpened():
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps > 0 and frame_count > 0:
            duration = frame_count / fps
        else:
            duration = 0.0
        cap.release()

        total_videos += 1
        total_frames += frame_count
        total_duration += duration
        all_durations.append(duration)
        all_frame_counts.append(frame_count)
        vids_in_sentence += 1

        # Classify translator vs user
        name_lower = vf.stem.lower()
        if "translator" in name_lower:
            translator_count += 1
        elif "user" in name_lower:
            user_count += 1
        else:
            unknown_count += 1

    if vids_in_sentence > 0:
        sentence_counts[label] = vids_in_sentence

num_sentences = len(sentence_counts)
vids_per_sent = list(sentence_counts.values())

log(f"Total videos:            {total_videos}")
log(f"Unique sentences:        {num_sentences}")
log(f"Translator videos:       {translator_count}")
log(f"User videos:             {user_count}")
log(f"Unknown-type videos:     {unknown_count}")
log(f"Total duration:          {total_duration:.1f} sec ({total_duration/60:.1f} min)")
log(f"Avg duration/video:      {np.mean(all_durations):.2f} sec (std={np.std(all_durations):.2f})")
log(f"Median duration/video:   {np.median(all_durations):.2f} sec")
log(f"Min/Max duration:        {np.min(all_durations):.2f} / {np.max(all_durations):.2f} sec")
log(f"Avg frames/video:        {np.mean(all_frame_counts):.1f} (std={np.std(all_frame_counts):.1f})")
log(f"Median frames/video:     {np.median(all_frame_counts):.1f}")
log(f"Min/Max frames:          {np.min(all_frame_counts)} / {np.max(all_frame_counts)}")
log(f"Videos/sentence min:     {min(vids_per_sent)}")
log(f"Videos/sentence max:     {max(vids_per_sent)}")
log(f"Videos/sentence avg:     {np.mean(vids_per_sent):.1f}")
log()


# ═══════════════════════════════════════════════════════════
# 2. TRANSLATOR vs USER ACCURACY SPLIT
# ═══════════════════════════════════════════════════════════
log("=" * 70)
log("2. LEAVE-ONE-OUT ACCURACY (All / Translator / User)")
log("=" * 70)

def cosine_sim(a, b):
    v1, v2 = a.flatten(), b.flatten()
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

# Load all matrices with type info
flat = []  # (label, name, mat, type)
for folder in sorted(MATRICES_DIR.iterdir()):
    if not folder.is_dir():
        continue
    label = folder.name
    for p in sorted(folder.glob("*.npy")):
        mat = np.load(p)
        name_lower = p.stem.lower()
        if "translator" in name_lower:
            vtype = "translator"
        elif "user" in name_lower:
            vtype = "user"
        else:
            vtype = "unknown"
        flat.append((label, p.stem, mat, vtype))

log(f"Total vectors loaded: {len(flat)}")
log(f"  Translator: {sum(1 for _,_,_,t in flat if t=='translator')}")
log(f"  User:       {sum(1 for _,_,_,t in flat if t=='user')}")
log(f"  Unknown:    {sum(1 for _,_,_,t in flat if t=='unknown')}")
log()

# Leave-one-out with type tracking
results = {"all": {"top1": 0, "top3": 0, "top5": 0, "total": 0},
           "translator": {"top1": 0, "top3": 0, "top5": 0, "total": 0},
           "user": {"top1": 0, "top3": 0, "top5": 0, "total": 0},
           "unknown": {"top1": 0, "top3": 0, "top5": 0, "total": 0}}

for idx in range(len(flat)):
    q_label, q_name, q_mat, q_type = flat[idx]
    
    scores = []
    for jdx in range(len(flat)):
        if idx == jdx:
            continue
        r_label, _, r_mat, _ = flat[jdx]
        sim = cosine_sim(q_mat, r_mat)
        scores.append((r_label, sim))
    
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Deduplicate: best score per label
    seen = set()
    ranked = []
    for lab, sc in scores:
        if lab not in seen:
            ranked.append(lab)
            seen.add(lab)
    
    for cat in ["all", q_type]:
        results[cat]["total"] += 1
        if ranked[0] == q_label:
            results[cat]["top1"] += 1
        if q_label in ranked[:3]:
            results[cat]["top3"] += 1
        if q_label in ranked[:5]:
            results[cat]["top5"] += 1

    if (idx + 1) % 50 == 0:
        log(f"  Evaluated {idx+1}/{len(flat)}...")

log()
for cat in ["all", "translator", "user", "unknown"]:
    t = results[cat]["total"]
    if t == 0:
        continue
    t1 = results[cat]["top1"] / t * 100
    t3 = results[cat]["top3"] / t * 100
    t5 = results[cat]["top5"] / t * 100
    log(f"  {cat.upper():12s}  (n={t:3d})  Top-1: {t1:5.1f}%  Top-3: {t3:5.1f}%  Top-5: {t5:5.1f}%")

log()


# ═══════════════════════════════════════════════════════════
# 3. LATENCY BENCHMARK
# ═══════════════════════════════════════════════════════════
log("=" * 70)
log("3. LATENCY BENCHMARK")
log("=" * 70)

# Pick a sample video for benchmarking
sample_video = None
for sentence_folder in sorted(VIDEOS_DIR.iterdir()):
    if not sentence_folder.is_dir():
        continue
    for vf in sorted(sentence_folder.iterdir()):
        if vf.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
            sample_video = vf
            break
    if sample_video:
        break

log(f"Sample video: {sample_video}")

import mediapipe as mp

N_FRAMES = 64
N_RUNS = 10  # average over 10 runs

# --- Stage 1: Frame read + sampling ---
stage1_times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(str(sample_video))
    total_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.round(np.linspace(0, total_frames_vid - 1, N_FRAMES)).astype(int)
    frames = []
    for i in range(total_frames_vid):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    # Pad if needed
    while len(frames) < N_FRAMES:
        frames.append(frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8))
    t1 = time.perf_counter()
    stage1_times.append((t1 - t0) * 1000)

avg_stage1 = np.mean(stage1_times)
log(f"Stage 1 - Frame read + sample ({N_RUNS} runs): {avg_stage1:.1f} ms (std={np.std(stage1_times):.1f})")

# --- Stage 2: MediaPipe extraction ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# Warm up
for frame in frames[:3]:
    hands.process(frame)

stage2_times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    feature_rows = []
    for frame in frames[:N_FRAMES]:
        result = hands.process(frame)
        left_hand = np.zeros(63)
        right_hand = np.zeros(63)
        if result.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                # Wrist normalization
                wrist = coords[0]
                centered = coords - wrist
                max_dist = np.max(np.linalg.norm(centered, axis=1))
                if max_dist > 0:
                    centered = centered / max_dist
                flat_coords = centered.flatten()  # 63
                label_hand = handedness.classification[0].label
                if label_hand == "Left":
                    left_hand = flat_coords
                else:
                    right_hand = flat_coords
        feature_rows.append(np.concatenate([left_hand, right_hand]))
    feature_matrix = np.array(feature_rows)  # (64, 126)
    t1 = time.perf_counter()
    stage2_times.append((t1 - t0) * 1000)

avg_stage2 = np.mean(stage2_times)
log(f"Stage 2 - MediaPipe extraction ({N_RUNS} runs): {avg_stage2:.1f} ms (std={np.std(stage2_times):.1f})")

hands.close()

# --- Stage 3: Cosine search ---
query_vec = feature_matrix.flatten()
all_ref_vecs = [f[2].flatten() for f in flat]
query_norm = np.linalg.norm(query_vec)

stage3_times = []
for _ in range(N_RUNS):
    t0 = time.perf_counter()
    scores = []
    for ref_vec in all_ref_vecs:
        ref_norm = np.linalg.norm(ref_vec)
        if query_norm == 0 or ref_norm == 0:
            scores.append(0.0)
        else:
            scores.append(float(np.dot(query_vec, ref_vec) / (query_norm * ref_norm)))
    # Sort + deduplicate
    paired = [(flat[i][0], scores[i]) for i in range(len(flat))]
    paired.sort(key=lambda x: x[1], reverse=True)
    t1 = time.perf_counter()
    stage3_times.append((t1 - t0) * 1000)

avg_stage3 = np.mean(stage3_times)
log(f"Stage 3 - Cosine search ({len(flat)} vecs, {N_RUNS} runs): {avg_stage3:.1f} ms (std={np.std(stage3_times):.1f})")

total_latency = avg_stage1 + avg_stage2 + avg_stage3
log(f"TOTAL per query: {total_latency:.1f} ms")
log()

# --- Hardware info ---
log("=" * 70)
log("4. HARDWARE INFO")
log("=" * 70)
log(f"Platform:   {platform.platform()}")
log(f"Processor:  {platform.processor()}")
log(f"Python:     {platform.python_version()}")
try:
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024**3)
    log(f"RAM:        {ram_gb:.1f} GB")
except ImportError:
    log("RAM:        (install psutil for RAM info)")
log()

# ═══════════════════════════════════════════════════════════
# 4. ACCURACY vs N (already computed, load from file)
# ═══════════════════════════════════════════════════════════
acc_file = Path("outputs/figures/paper_figures") / "accuracy_vs_n_results.txt"
if acc_file.exists():
    log("=" * 70)
    log("5. ACCURACY vs N (from previous run)")
    log("=" * 70)
    log(acc_file.read_text())

# Save all
OUTPUT.write_text("\n".join(lines), encoding="utf-8")
log(f"\nAll stats saved to: {OUTPUT}")
