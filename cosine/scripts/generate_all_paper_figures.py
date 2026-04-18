"""
Generate all 6 paper figures for Springer LNCS publication.

  Fig. 1  System architecture pipeline
  Fig. 2  MediaPipe 21-landmark hand model
  Fig. 3  Feature matrix heatmap (2 examples: same vs. different sentence)
  Fig. 4  State machine diagram (IDLE → RECORDING → COOLDOWN → MATCHING)
  Fig. 5  Bar chart: accuracy vs. frame count N  (N = 16, 32, 64, 128)
  Fig. 6  Real-time system screenshot placeholder (annotated)

Usage:
    python generate_all_paper_figures.py
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ArrowStyle
from pathlib import Path
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ── Config ─────────────────────────────────────────────
MATRICES_DIR = Path("data/processed/matrices")
OUTPUT_DIR = Path("outputs/figures/paper_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

SENTENCE_A = "Azərbaycanın paytaxtı Bakı şəhəridir"
SENTENCE_B = "Bu gün hava çox soyuqdur"
SENTENCE_C = "Bayrağımız 3 rənglidir_ göy, qırmızı, yaşıl"

DPI = 300
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
})

# ── Helpers ────────────────────────────────────────────
def load_sentence_matrices(label):
    folder = MATRICES_DIR / label
    if not folder.exists():
        return {}
    return {p.stem: np.load(p) for p in sorted(folder.glob("*.npy"))}

def cosine_sim(a, b):
    v1, v2 = a.flatten(), b.flatten()
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))

def load_all_matrices():
    db = {}
    for folder in sorted(MATRICES_DIR.iterdir()):
        if folder.is_dir():
            mats = {p.stem: np.load(p) for p in sorted(folder.glob("*.npy"))}
            if mats:
                db[folder.name] = mats
    return db

def resample_matrix(mat, target_n):
    """Resample a 64×126 matrix to target_n×126 by uniform index selection."""
    src_n = mat.shape[0]
    if target_n == src_n:
        return mat
    indices = np.round(np.linspace(0, src_n - 1, target_n)).astype(int)
    return mat[indices]


# ══════════════════════════════════════════════════════════
# FIGURE 1: System Architecture Pipeline
# ══════════════════════════════════════════════════════════
def fig1_system_architecture():
    print("Fig 1: System architecture pipeline...", flush=True)

    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.2, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Box style
    box_kw = dict(boxstyle="round,pad=0.3", edgecolor="#333333", linewidth=1.5)
    text_kw = dict(ha="center", va="center", fontsize=9, fontweight="bold", family="serif")

    # Define boxes: (x, y, width, height, label, color)
    boxes = [
        (0.0, 0.5, 1.6, 0.9, "Video\nInput",           "#E3F2FD"),
        (2.2, 0.5, 1.6, 0.9, "Frame\nSampling\n(N=64)", "#E8F5E9"),
        (4.4, 0.5, 1.6, 0.9, "MediaPipe\nHands\n(126D/frame)", "#FFF3E0"),
        (6.6, 0.5, 1.6, 0.9, "Feature\nMatrix\n(64×126)", "#F3E5F5"),
        (8.8, 0.5, 1.6, 0.9, "Cosine\nSimilarity\nSearch", "#FFEBEE"),
    ]

    # Draw boxes
    for (x, y, w, h, label, color) in boxes:
        fancy = FancyBboxPatch((x, y), w, h, **box_kw, facecolor=color)
        ax.add_patch(fancy)
        ax.text(x + w / 2, y + h / 2, label, **text_kw)

    # Draw arrows between boxes
    arrow_kw = dict(arrowstyle="-|>", color="#555555", linewidth=1.8,
                    mutation_scale=15, connectionstyle="arc3,rad=0")
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]      # right edge
        x2 = boxes[i + 1][0]                 # left edge of next
        y_mid = boxes[i][1] + boxes[i][3] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle="-|>", color="#555555",
                                    lw=1.8, mutation_scale=15))

    # Result box below the last box
    rx, ry, rw, rh = 8.8, -0.8, 1.6, 0.7
    fancy = FancyBboxPatch((rx, ry), rw, rh, **box_kw, facecolor="#C8E6C9")
    ax.add_patch(fancy)
    ax.text(rx + rw / 2, ry + rh / 2, "Top-K\nResults", **text_kw)
    # Vertical arrow from search to results
    ax.annotate("", xy=(rx + rw / 2, ry + rh), xytext=(rx + rw / 2, boxes[-1][1]),
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.8, mutation_scale=15))

    # Database symbol next to cosine search
    db_x, db_y = 6.6, -0.8
    fancy = FancyBboxPatch((db_x, db_y), 1.6, 0.7, **box_kw, facecolor="#E0E0E0")
    ax.add_patch(fancy)
    ax.text(db_x + 0.8, db_y + 0.35, "Reference\nDatabase",
            ha="center", va="center", fontsize=8, fontweight="bold", family="serif")
    # Arrow from database to cosine search
    ax.annotate("", xy=(boxes[-1][0], boxes[-1][1]),
                xytext=(db_x + 1.6, db_y + 0.7),
                arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.3,
                                mutation_scale=12, linestyle="--"))

    # Labels for offline/online
    ax.text(5.0, 2.0, "Online: webcam query", fontsize=8, ha="center",
            style="italic", color="#1565C0")
    ax.text(7.4, -1.25, "Offline: pre-computed", fontsize=8, ha="center",
            style="italic", color="#666666")

    plt.savefig(OUTPUT_DIR / "fig1_system_architecture.png")
    plt.close()
    print("  Saved fig1_system_architecture.png", flush=True)


# ══════════════════════════════════════════════════════════
# FIGURE 2: MediaPipe 21-Landmark Hand Model
# ══════════════════════════════════════════════════════════
def fig2_mediapipe_hand():
    print("Fig 2: MediaPipe 21-landmark hand model...", flush=True)

    # MediaPipe hand landmark coordinates (approximate positions on a right hand)
    # Landmark positions in 2D (schematic)
    landmarks = {
        0:  (0.50, 0.05),   # WRIST
        1:  (0.38, 0.20),   # THUMB_CMC
        2:  (0.28, 0.32),   # THUMB_MCP
        3:  (0.20, 0.44),   # THUMB_IP
        4:  (0.13, 0.54),   # THUMB_TIP
        5:  (0.32, 0.52),   # INDEX_MCP
        6:  (0.30, 0.68),   # INDEX_PIP
        7:  (0.29, 0.80),   # INDEX_DIP
        8:  (0.28, 0.92),   # INDEX_TIP
        9:  (0.44, 0.55),   # MIDDLE_MCP
        10: (0.43, 0.72),   # MIDDLE_PIP
        11: (0.42, 0.84),   # MIDDLE_DIP
        12: (0.42, 0.96),   # MIDDLE_TIP
        13: (0.56, 0.52),   # RING_MCP
        14: (0.57, 0.67),   # RING_PIP
        15: (0.57, 0.78),   # RING_DIP
        16: (0.58, 0.89),   # RING_TIP
        17: (0.67, 0.46),   # PINKY_MCP
        18: (0.69, 0.58),   # PINKY_PIP
        19: (0.70, 0.67),   # PINKY_DIP
        20: (0.71, 0.76),   # PINKY_TIP
    }

    # Connections (MediaPipe hand topology)
    connections = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
        # Palm connections
        (5, 9), (9, 13), (13, 17),
    ]

    # Finger names and label positions
    finger_labels = [
        ("Thumb", 0.10, 0.58),
        ("Index", 0.20, 0.96),
        ("Middle", 0.42, 1.01),
        ("Ring", 0.62, 0.93),
        ("Pinky", 0.78, 0.80),
    ]

    # Joint type names
    joint_names = {
        0: "Wrist", 4: "Tip", 8: "Tip", 12: "Tip", 16: "Tip", 20: "Tip",
        2: "MCP", 6: "PIP", 10: "PIP", 14: "PIP", 18: "PIP",
        3: "IP", 7: "DIP", 11: "DIP", 15: "DIP", 19: "DIP",
    }

    fig, ax = plt.subplots(figsize=(5, 6.5))
    ax.set_xlim(-0.05, 0.95)
    ax.set_ylim(-0.05, 1.10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw connections
    for (i, j) in connections:
        x1, y1 = landmarks[i]
        x2, y2 = landmarks[j]
        ax.plot([x1, x2], [y1, y2], color="#78909C", linewidth=2.5, zorder=1)

    # Color code by finger
    finger_colors = {
        "wrist": "#E53935",
        "thumb": "#FB8C00",
        "index": "#43A047",
        "middle": "#1E88E5",
        "ring": "#8E24AA",
        "pinky": "#00ACC1",
    }

    def get_finger(idx):
        if idx == 0: return "wrist"
        if idx <= 4: return "thumb"
        if idx <= 8: return "index"
        if idx <= 12: return "middle"
        if idx <= 16: return "ring"
        return "pinky"

    # Draw landmarks
    for idx, (x, y) in landmarks.items():
        color = finger_colors[get_finger(idx)]
        ax.scatter(x, y, s=180, c=color, edgecolors="white", linewidth=1.2, zorder=3)
        ax.annotate(str(idx), (x, y), fontsize=7, fontweight="bold",
                    ha="center", va="center", color="white", zorder=4)

    # Finger labels
    for name, lx, ly in finger_labels:
        ax.text(lx, ly, name, fontsize=8, ha="center", style="italic", color="#555555")

    # Legend: features info
    info_text = "21 landmarks × 3 coords (x, y, z) × 2 hands = 126 features/frame"
    ax.text(0.45, -0.03, info_text, fontsize=8, ha="center", va="top",
            style="italic", color="#333333")

    plt.savefig(OUTPUT_DIR / "fig2_mediapipe_hand.png")
    plt.close()
    print("  Saved fig2_mediapipe_hand.png", flush=True)


# ══════════════════════════════════════════════════════════
# FIGURE 3: Feature Matrix Heatmaps (same + different sentence)
# ══════════════════════════════════════════════════════════
def fig3_feature_heatmaps():
    print("Fig 3: Feature matrix heatmaps...", flush=True)
    mats_a = load_sentence_matrices(SENTENCE_A)
    mats_b = load_sentence_matrices(SENTENCE_B)

    names_a = list(mats_a.keys())
    if len(names_a) < 2 or not mats_b:
        print("  Not enough data, skipping.")
        return

    m1 = mats_a[names_a[0]]  # Same sentence, signer 1
    m2 = mats_a[names_a[1]]  # Same sentence, signer 2
    m3 = list(mats_b.values())[0]  # Different sentence

    sim_same = cosine_sim(m1, m2)
    sim_diff = cosine_sim(m1, m3)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    labels_sub = [
        f"(a) Sentence A – Signer 1",
        f"(b) Sentence A – Signer 2",
        f"(c) Sentence B – Signer 1",
    ]

    for ax, mat, lab in zip(axes, [m1, m2, m3], labels_sub):
        im = ax.imshow(mat.T, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Feature dimension")
        ax.text(0.5, -0.22, lab, transform=ax.transAxes, fontsize=9,
                ha="center", va="top")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Normalized feature value")

    # Add similarity annotations
    fig.text(0.28, 0.01, f"cos(a,b) = {sim_same:.3f}", fontsize=9, ha="center",
             color="#1565C0", fontweight="bold")
    fig.text(0.68, 0.01, f"cos(a,c) = {sim_diff:.3f}", fontsize=9, ha="center",
             color="#BF360C", fontweight="bold")

    plt.savefig(OUTPUT_DIR / "fig3_feature_heatmaps.png")
    plt.close()
    print(f"  Saved. cos(same)={sim_same:.3f}, cos(diff)={sim_diff:.3f}", flush=True)


# ══════════════════════════════════════════════════════════
# FIGURE 4: State Machine Diagram
# ══════════════════════════════════════════════════════════
def fig4_state_machine():
    print("Fig 4: State machine diagram...", flush=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-1.5, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")

    # State positions: (cx, cy)
    states = {
        "IDLE":      (1.0, 1.0),
        "RECORDING": (4.0, 1.0),
        "COOLDOWN":  (7.0, 1.0),
        "MATCHING":  (5.5, -0.5),
    }

    # State colors
    colors = {
        "IDLE": "#E3F2FD",
        "RECORDING": "#C8E6C9",
        "COOLDOWN": "#FFF3E0",
        "MATCHING": "#F3E5F5",
    }

    border_colors = {
        "IDLE": "#1565C0",
        "RECORDING": "#2E7D32",
        "COOLDOWN": "#E65100",
        "MATCHING": "#6A1B9A",
    }

    box_w, box_h = 2.0, 0.85

    # Draw state boxes
    for name, (cx, cy) in states.items():
        x = cx - box_w / 2
        y = cy - box_h / 2
        fancy = FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.15",
            facecolor=colors[name],
            edgecolor=border_colors[name],
            linewidth=2.0,
        )
        ax.add_patch(fancy)
        ax.text(cx, cy, name, ha="center", va="center",
                fontsize=11, fontweight="bold", color=border_colors[name])

    # Transitions with labels
    def draw_arrow(from_state, to_state, label, offset=(0, 0), curve=0):
        fx, fy = states[from_state]
        tx, ty = states[to_state]

        # Determine start/end points at box edges
        if fx < tx:
            sx = fx + box_w / 2
            ex = tx - box_w / 2
        elif fx > tx:
            sx = fx - box_w / 2
            ex = tx + box_w / 2
        else:
            sx, ex = fx, tx

        if fy == ty:
            sy, ey = fy, ty
        elif fy > ty:
            sy = fy - box_h / 2
            ey = ty + box_h / 2
        else:
            sy = fy + box_h / 2
            ey = ty - box_h / 2

        conn = f"arc3,rad={curve}" if curve != 0 else "arc3,rad=0"
        ax.annotate("", xy=(ex, ey), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle="-|>", color="#555555",
                                    lw=1.5, mutation_scale=14,
                                    connectionstyle=conn))
        # Label
        mx = (sx + ex) / 2 + offset[0]
        my = (sy + ey) / 2 + offset[1]
        ax.text(mx, my, label, fontsize=7.5, ha="center", va="center",
                color="#333333", style="italic",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9))

    # IDLE → RECORDING
    draw_arrow("IDLE", "RECORDING", "hands\ndetected", offset=(0, 0.55))

    # RECORDING → COOLDOWN
    draw_arrow("RECORDING", "COOLDOWN", "hands\nlost", offset=(0, 0.55))

    # COOLDOWN → RECORDING (hands come back - curved above)
    fx, fy = states["COOLDOWN"]
    tx, ty = states["RECORDING"]
    ax.annotate("", xy=(tx + box_w / 2, ty + box_h / 2),
                xytext=(fx - box_w / 2, fy + box_h / 2),
                arrowprops=dict(arrowstyle="-|>", color="#2E7D32",
                                lw=1.3, mutation_scale=12,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(5.5, 2.3, "hands return", fontsize=7.5, ha="center",
            color="#2E7D32", style="italic",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="#C8E6C9", alpha=0.9))

    # COOLDOWN → MATCHING
    draw_arrow("COOLDOWN", "MATCHING", "timeout\n(2 sec)", offset=(-0.1, 0.0), curve=0)

    # MATCHING → IDLE
    draw_arrow("MATCHING", "IDLE", "display\nresults", offset=(0.0, -0.6), curve=0)

    # Start indicator (small filled circle before IDLE)
    ax.scatter(-0.2, 1.0, s=80, color="#333333", zorder=5)
    ax.annotate("", xy=(states["IDLE"][0] - box_w / 2, 1.0), xytext=(0.0, 1.0),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.5, mutation_scale=14))

    plt.savefig(OUTPUT_DIR / "fig4_state_machine.png")
    plt.close()
    print("  Saved fig4_state_machine.png", flush=True)


# ══════════════════════════════════════════════════════════
# FIGURE 5: Bar Chart – Accuracy vs Frame Count N
# ══════════════════════════════════════════════════════════
def fig5_accuracy_vs_n():
    print("Fig 5: Accuracy vs. frame count N...", flush=True)
    db_full = load_all_matrices()  # All stored at N=64

    N_values = [16, 32, 64, 128]
    results = {}

    for N in N_values:
        print(f"  Evaluating N={N}...", flush=True)
        # Build flat list with resampled matrices
        flat = []
        for label, mats in db_full.items():
            for name, mat in mats.items():
                resampled = resample_matrix(mat, N)
                flat.append((label, resampled))

        top1 = top3 = top5 = 0
        total = len(flat)

        for idx in range(total):
            q_label, q_mat = flat[idx]
            scores = []
            for jdx in range(total):
                if idx == jdx:
                    continue
                r_label, r_mat = flat[jdx]
                sim = cosine_sim(q_mat, r_mat)
                scores.append((r_label, sim))
            scores.sort(key=lambda x: x[1], reverse=True)

            # Deduplicate: best per label
            seen = set()
            ranked = []
            for lab, sc in scores:
                if lab not in seen:
                    ranked.append(lab)
                    seen.add(lab)

            if ranked[0] == q_label:
                top1 += 1
            if q_label in ranked[:3]:
                top3 += 1
            if q_label in ranked[:5]:
                top5 += 1

        results[N] = (top1 / total * 100, top3 / total * 100, top5 / total * 100)
        print(f"    Top-1: {results[N][0]:.1f}%  Top-3: {results[N][1]:.1f}%  Top-5: {results[N][2]:.1f}%", flush=True)

    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(N_values))
    width = 0.25  # bar width

    t1_vals = [results[n][0] for n in N_values]
    t3_vals = [results[n][1] for n in N_values]
    t5_vals = [results[n][2] for n in N_values]

    bars1 = ax.bar(x - width, t1_vals, width, label="Top-1", color="#1976D2", edgecolor="white")
    bars2 = ax.bar(x,         t3_vals, width, label="Top-3", color="#388E3C", edgecolor="white")
    bars3 = ax.bar(x + width, t5_vals, width, label="Top-5", color="#F57C00", edgecolor="white")

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xlabel("Number of Sampled Frames (N)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in N_values])
    ax.set_ylim(0, max(max(t1_vals), max(t3_vals), max(t5_vals)) + 10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(OUTPUT_DIR / "fig5_accuracy_vs_n.png")
    plt.close()
    print("  Saved fig5_accuracy_vs_n.png", flush=True)

    # Save data
    with open(OUTPUT_DIR / "accuracy_vs_n_results.txt", "w") as f:
        f.write("N\tTop-1\tTop-3\tTop-5\n")
        for n in N_values:
            f.write(f"{n}\t{results[n][0]:.2f}\t{results[n][1]:.2f}\t{results[n][2]:.2f}\n")
    print("  Results saved to accuracy_vs_n_results.txt", flush=True)


# ══════════════════════════════════════════════════════════
# FIGURE 6: Real-Time System Screenshot (simulated)
# ══════════════════════════════════════════════════════════
def fig6_realtime_screenshot():
    print("Fig 6: Real-time system screenshot (simulated layout)...", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 480)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.invert_yaxis()

    # Dark background
    ax.add_patch(plt.Rectangle((0, 0), 640, 480, facecolor="#1A1A2E", zorder=0))

    # Camera frame area
    ax.add_patch(plt.Rectangle((10, 10), 620, 380, facecolor="#0F3460",
                                edgecolor="#16213E", linewidth=2, zorder=1))
    ax.text(320, 200, "WEBCAM FEED\n(with MediaPipe landmarks overlay)",
            ha="center", va="center", fontsize=13, color="#E0E0E0", zorder=2)

    # Hand landmark placeholder
    # Draw a simple hand skeleton inside the frame
    hand_x, hand_y = 320, 180
    hand_pts = [
        (hand_x, hand_y + 50),      # wrist
        (hand_x - 30, hand_y + 20), # thumb
        (hand_x - 20, hand_y - 30), # index
        (hand_x, hand_y - 40),      # middle
        (hand_x + 15, hand_y - 25), # ring
        (hand_x + 30, hand_y - 10), # pinky
    ]
    for pt in hand_pts:
        ax.scatter(*pt, s=30, c="#00FF00", zorder=4)
    # Connections from wrist
    wrist = hand_pts[0]
    for pt in hand_pts[1:]:
        ax.plot([wrist[0], pt[0]], [wrist[1], pt[1]], color="#00FF00",
                linewidth=1.5, alpha=0.7, zorder=3)

    # Status bar at top
    ax.add_patch(plt.Rectangle((10, 10), 620, 35, facecolor="#00000080", zorder=5))
    ax.text(30, 28, "● RECORDING", fontsize=11, color="#4CAF50",
            fontweight="bold", va="center", zorder=6, family="monospace")
    ax.text(200, 28, "53 frames", fontsize=10, color="#FFFFFF",
            va="center", zorder=6, family="monospace")

    # Results overlay at bottom
    ax.add_patch(plt.Rectangle((10, 400), 620, 70, facecolor="#000000B0",
                                edgecolor="#333333", linewidth=1, zorder=5))

    results_text = [
        ("1.", "Azərbaycanın paytaxtı Bakı...", "0.847", "#4CAF50"),
        ("2.", "Bayrağımız 3 rənglidir...", "0.723", "#FFC107"),
        ("3.", "İtaliya çox gözəldir...", "0.691", "#FF9800"),
    ]

    for i, (rank, sentence, score, color) in enumerate(results_text):
        y_pos = 415 + i * 18
        ax.text(30, y_pos, rank, fontsize=9, color=color, fontweight="bold",
                va="center", zorder=6, family="monospace")
        ax.text(55, y_pos, sentence, fontsize=9, color="#FFFFFF",
                va="center", zorder=6)
        ax.text(580, y_pos, score, fontsize=9, color=color, fontweight="bold",
                va="center", ha="right", zorder=6, family="monospace")

    # Controls hint
    ax.text(320, 478, "q: quit  |  r: reset", fontsize=8, ha="center",
            color="#888888", va="bottom", zorder=6)

    plt.savefig(OUTPUT_DIR / "fig6_realtime_screenshot.png", facecolor="#1A1A2E")
    plt.close()
    print("  Saved fig6_realtime_screenshot.png", flush=True)


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Generating all 6 paper figures")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("=" * 60, flush=True)

    fig1_system_architecture()
    fig2_mediapipe_hand()
    fig3_feature_heatmaps()
    fig4_state_machine()
    fig5_accuracy_vs_n()      # This one takes a while (leave-one-out × 4 N values)
    fig6_realtime_screenshot()

    print("\n" + "=" * 60)
    print("All 6 figures generated:")
    print("  Fig 1: fig1_system_architecture.png")
    print("  Fig 2: fig2_mediapipe_hand.png")
    print("  Fig 3: fig3_feature_heatmaps.png")
    print("  Fig 4: fig4_state_machine.png")
    print("  Fig 5: fig5_accuracy_vs_n.png")
    print("  Fig 6: fig6_realtime_screenshot.png")
    print(f"\nAll saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
