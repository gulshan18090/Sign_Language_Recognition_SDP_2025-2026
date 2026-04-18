"""
Generate publication-ready figures for the research paper.

Produces 7 figures saved to paper_figures/ :
  1. feature_matrix_heatmap.png       – Side-by-side heatmaps (same sentence, 2 signers)
  2. feature_matrix_diff_sentence.png – Heatmaps for different sentences (contrast)
  3. similarity_matrix.png            – Cross-video cosine similarity matrix (heatmap)
  4. score_distribution.png           – Histogram: same-sentence vs. different-sentence scores
  5. frame_cosine_curve.png           – Per-frame cosine similarity for matched vs. unmatched
  6. topk_accuracy_bar.png            – Top-1/3/5 accuracy across different N_FRAMES
  7. hand_detection_ratio.png         – Bar chart of hand detection ratio per video

Usage:
    python generate_paper_figures.py
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.spatial.distance import cosine
from itertools import combinations

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# ── Config ─────────────────────────────────────────────
MATRICES_DIR = Path("data/processed/matrices")
OUTPUT_DIR = Path("outputs/figures/paper_figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Pick sentences for visualization
SENTENCE_A = "Azərbaycanın paytaxtı Bakı şəhəridir"   # 5 videos
SENTENCE_B = "Bu gün hava çox soyuqdur"                 # different sentence
SENTENCE_C = "Bayrağımız 3 rənglidir_ göy, qırmızı, yaşıl"

DPI = 300
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})
# ───────────────────────────────────────────────────────


def load_sentence_matrices(sentence_label: str):
    """Load all .npy matrices for a sentence."""
    folder = MATRICES_DIR / sentence_label
    if not folder.exists():
        print(f"  WARNING: folder not found: {folder}")
        return {}
    result = {}
    for p in sorted(folder.glob("*.npy")):
        result[p.stem] = np.load(p)
    return result


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    v1, v2 = a.flatten(), b.flatten()
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(1.0 - cosine(v1, v2))


def frame_wise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-frame cosine similarity (length = min of both)."""
    n = min(a.shape[0], b.shape[0])
    sims = []
    for i in range(n):
        n1, n2 = np.linalg.norm(a[i]), np.linalg.norm(b[i])
        if n1 == 0 or n2 == 0:
            sims.append(0.0)
        else:
            sims.append(1.0 - cosine(a[i], b[i]))
    return np.array(sims)


def load_all_matrices():
    """Load all matrices organized by sentence label."""
    db = {}
    for folder in sorted(MATRICES_DIR.iterdir()):
        if folder.is_dir():
            label = folder.name
            matrices = {}
            for p in sorted(folder.glob("*.npy")):
                matrices[p.stem] = np.load(p)
            if matrices:
                db[label] = matrices
    return db


# ══════════════════════════════════════════════════════════
# FIGURE 1: Feature Matrix Heatmaps – Same Sentence
# ══════════════════════════════════════════════════════════
def fig1_heatmap_same_sentence():
    print("Figure 1: Feature matrix heatmaps (same sentence)...", flush=True)
    mats = load_sentence_matrices(SENTENCE_A)
    names = list(mats.keys())
    if len(names) < 2:
        print("  Not enough videos, skipping.")
        return

    m1, m2 = mats[names[0]], mats[names[1]]
    sim = cosine_sim(m1, m2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    for ax, mat, name in zip(axes, [m1, m2], names[:2]):
        im = ax.imshow(mat.T, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Feature dimension")
    fig.colorbar(im, ax=axes, shrink=0.8, label="Feature value")
    plt.savefig(OUTPUT_DIR / "feature_matrix_heatmap.png")
    plt.close()
    print(f"  Saved. Cosine sim = {sim:.4f}")


# ══════════════════════════════════════════════════════════
# FIGURE 2: Feature Matrix Heatmaps – Different Sentences
# ══════════════════════════════════════════════════════════
def fig2_heatmap_diff_sentence():
    print("Figure 2: Feature matrix heatmaps (different sentences)...", flush=True)
    mats_a = load_sentence_matrices(SENTENCE_A)
    mats_b = load_sentence_matrices(SENTENCE_B)
    if not mats_a or not mats_b:
        print("  Missing data, skipping.")
        return

    m1 = list(mats_a.values())[0]
    m2 = list(mats_b.values())[0]
    sim = cosine_sim(m1, m2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    labels = [SENTENCE_A[:40] + "...", SENTENCE_B[:40] + "..."]
    for ax, mat, lab in zip(axes, [m1, m2], labels):
        im = ax.imshow(mat.T, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Feature dimension")

    fig.colorbar(im, ax=axes, shrink=0.8, label="Feature value")
    plt.savefig(OUTPUT_DIR / "feature_matrix_diff_sentence.png")
    plt.close()
    print(f"  Saved. Cosine sim = {sim:.4f}")


# ══════════════════════════════════════════════════════════
# FIGURE 3: Cross-Video Cosine Similarity Matrix
# ══════════════════════════════════════════════════════════
def fig3_similarity_matrix():
    print("Figure 3: Cross-video similarity matrix...", flush=True)
    # Pick 3 sentences, get all their videos
    sentences = [SENTENCE_A, SENTENCE_B, SENTENCE_C]
    all_labels = []
    all_mats = []

    for sent in sentences:
        mats = load_sentence_matrices(sent)
        for name, mat in mats.items():
            short_sent = sent[:20] + "..."
            short_name = name.replace("translator_video_", "T").replace("user_video_", "U")[:12]
            all_labels.append(f"{short_sent}\n{short_name}")
            all_mats.append(mat)

    n = len(all_mats)
    if n < 3:
        print("  Not enough data, skipping.")
        return

    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = cosine_sim(all_mats[i], all_mats[j])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(n))
    ax.set_xticklabels(all_labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(all_labels, fontsize=6)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if sim_matrix[i, j] < 0.4 else "black"
            ax.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=5.5, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Cosine Similarity")
    plt.savefig(OUTPUT_DIR / "similarity_matrix.png")
    plt.close()
    print(f"  Saved. {n}×{n} matrix.")


# ══════════════════════════════════════════════════════════
# FIGURE 4: Score Distribution – Same vs. Different
# ══════════════════════════════════════════════════════════
def fig4_score_distribution():
    print("Figure 4: Score distribution (same vs. different)...", flush=True)
    db = load_all_matrices()
    labels = list(db.keys())

    same_scores = []
    diff_scores = []

    # Flatten: list of (label, matrix)
    flat = []
    for label, mats in db.items():
        for name, mat in mats.items():
            flat.append((label, mat))

    print(f"  Computing pairwise scores for {len(flat)} videos...", flush=True)

    # Sample to keep it tractable: all same-sentence + random different
    import random
    random.seed(42)

    # Same-sentence pairs
    for label, mats in db.items():
        mat_list = list(mats.values())
        for i in range(len(mat_list)):
            for j in range(i + 1, len(mat_list)):
                same_scores.append(cosine_sim(mat_list[i], mat_list[j]))

    # Different-sentence pairs (sample ~500)
    diff_pairs = []
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i][0] != flat[j][0]:
                diff_pairs.append((i, j))
    sampled = random.sample(diff_pairs, min(500, len(diff_pairs)))
    for i, j in sampled:
        diff_scores.append(cosine_sim(flat[i][1], flat[j][1]))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 40)
    ax.hist(same_scores, bins=bins, alpha=0.7, label=f"Same sentence (n={len(same_scores)})",
            color="#2196F3", edgecolor="white", linewidth=0.5)
    ax.hist(diff_scores, bins=bins, alpha=0.7, label=f"Different sentence (n={len(diff_scores)})",
            color="#FF5722", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.axvline(x=np.mean(same_scores), color="#1565C0", linestyle="--", linewidth=1,
               label=f"Same mean={np.mean(same_scores):.3f}")
    ax.axvline(x=np.mean(diff_scores), color="#BF360C", linestyle="--", linewidth=1,
               label=f"Diff mean={np.mean(diff_scores):.3f}")
    ax.legend(fontsize=8)

    plt.savefig(OUTPUT_DIR / "score_distribution.png")
    plt.close()
    print(f"  Saved. Same: mean={np.mean(same_scores):.3f}, Diff: mean={np.mean(diff_scores):.3f}")


# ══════════════════════════════════════════════════════════
# FIGURE 5: Frame-Wise Cosine Similarity Curves
# ══════════════════════════════════════════════════════════
def fig5_framewise_curves():
    print("Figure 5: Frame-wise cosine similarity curves...", flush=True)
    mats_a = load_sentence_matrices(SENTENCE_A)
    mats_b = load_sentence_matrices(SENTENCE_B)
    if len(mats_a) < 2 or not mats_b:
        print("  Not enough data, skipping.")
        return

    names_a = list(mats_a.keys())
    m_same_1 = mats_a[names_a[0]]
    m_same_2 = mats_a[names_a[1]]
    m_diff = list(mats_b.values())[0]

    curve_same = frame_wise_cosine(m_same_1, m_same_2)
    curve_diff = frame_wise_cosine(m_same_1, m_diff)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    frames = np.arange(len(curve_same))
    ax.plot(frames, curve_same, color="#2196F3", linewidth=1.5, alpha=0.9,
            label=f"Same sentence (avg={np.mean(curve_same):.3f})")
    ax.plot(frames[:len(curve_diff)], curve_diff, color="#FF5722", linewidth=1.5, alpha=0.9,
            label=f"Different sentence (avg={np.mean(curve_diff):.3f})")
    ax.fill_between(frames, curve_same, alpha=0.15, color="#2196F3")
    ax.fill_between(frames[:len(curve_diff)], curve_diff, alpha=0.15, color="#FF5722")

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Cosine Similarity")
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.savefig(OUTPUT_DIR / "frame_cosine_curve.png")
    plt.close()
    print("  Saved.")


# ══════════════════════════════════════════════════════════
# FIGURE 6: Top-K Accuracy vs N_FRAMES (Leave-One-Out)
# ══════════════════════════════════════════════════════════
def fig6_topk_accuracy():
    print("Figure 6: Top-K accuracy bar chart (N=64)...", flush=True)
    db = load_all_matrices()

    # Leave-one-out evaluation
    flat = []
    for label, mats in db.items():
        for name, mat in mats.items():
            flat.append((label, name, mat))

    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    total = len(flat)

    for idx, (query_label, query_name, query_mat) in enumerate(flat):
        scores = []
        for jdx, (ref_label, ref_name, ref_mat) in enumerate(flat):
            if idx == jdx:
                continue
            sim = cosine_sim(query_mat, ref_mat)
            scores.append((ref_label, sim))

        # Rank by score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate: best score per label
        seen = set()
        ranked_labels = []
        for lab, sc in scores:
            if lab not in seen:
                ranked_labels.append(lab)
                seen.add(lab)

        if ranked_labels[0] == query_label:
            top1_correct += 1
        if query_label in ranked_labels[:3]:
            top3_correct += 1
        if query_label in ranked_labels[:5]:
            top5_correct += 1

    top1_acc = top1_correct / total * 100
    top3_acc = top3_correct / total * 100
    top5_acc = top5_correct / total * 100

    print(f"  Top-1: {top1_acc:.1f}%  Top-3: {top3_acc:.1f}%  Top-5: {top5_acc:.1f}%")

    fig, ax = plt.subplots(figsize=(5, 4))
    metrics = ["Top-1", "Top-3", "Top-5"]
    values = [top1_acc, top3_acc, top5_acc]
    colors = ["#1976D2", "#388E3C", "#F57C00"]
    bars = ax.bar(metrics, values, color=colors, edgecolor="white", width=0.5)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)

    plt.savefig(OUTPUT_DIR / "topk_accuracy_bar.png")
    plt.close()
    print("  Saved.")

    # Save numbers to a text file for the paper
    with open(OUTPUT_DIR / "accuracy_results.txt", "w") as f:
        f.write(f"Leave-One-Out Evaluation (N=64 frames, {total} videos, {len(db)} sentences)\n")
        f.write(f"Top-1 Accuracy: {top1_acc:.2f}%  ({top1_correct}/{total})\n")
        f.write(f"Top-3 Accuracy: {top3_acc:.2f}%  ({top3_correct}/{total})\n")
        f.write(f"Top-5 Accuracy: {top5_acc:.2f}%  ({top5_correct}/{total})\n")
    print(f"  Results also saved to {OUTPUT_DIR}/accuracy_results.txt")


# ══════════════════════════════════════════════════════════
# FIGURE 7: Per-Sentence Accuracy Breakdown (horizontal bar)
# ══════════════════════════════════════════════════════════
def fig7_per_sentence_accuracy():
    print("Figure 7: Per-sentence top-1 accuracy breakdown...", flush=True)
    db = load_all_matrices()

    flat = []
    for label, mats in db.items():
        for name, mat in mats.items():
            flat.append((label, name, mat))

    # Per-sentence correct counts
    sentence_correct = {}
    sentence_total = {}

    for idx, (query_label, query_name, query_mat) in enumerate(flat):
        sentence_total[query_label] = sentence_total.get(query_label, 0) + 1

        scores = []
        for jdx, (ref_label, ref_name, ref_mat) in enumerate(flat):
            if idx == jdx:
                continue
            sim = cosine_sim(query_mat, ref_mat)
            scores.append((ref_label, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        seen = set()
        ranked_labels = []
        for lab, sc in scores:
            if lab not in seen:
                ranked_labels.append(lab)
                seen.add(lab)

        if ranked_labels[0] == query_label:
            sentence_correct[query_label] = sentence_correct.get(query_label, 0) + 1

    # Compute per-sentence accuracy
    accs = []
    for label in sorted(db.keys()):
        correct = sentence_correct.get(label, 0)
        total = sentence_total.get(label, 1)
        accs.append((label[:35] + ("..." if len(label) > 35 else ""), correct / total * 100, total))

    # Sort by accuracy
    accs.sort(key=lambda x: x[1])

    # Show top 20 and bottom 20 if too many
    if len(accs) > 30:
        display = accs[:10] + [("...", 0, 0)] + accs[-10:]
    else:
        display = accs

    fig, ax = plt.subplots(figsize=(8, max(4, len(display) * 0.28)))
    labels = [d[0] for d in display]
    vals = [d[1] for d in display]
    colors = ["#EF5350" if v < 50 else "#FFA726" if v < 80 else "#66BB6A" for v in vals]

    ax.barh(range(len(labels)), vals, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Top-1 Accuracy (%)")
    ax.set_xlim(0, 105)
    ax.axvline(x=50, color="gray", linestyle=":", alpha=0.5)
    ax.grid(axis="x", alpha=0.3)

    plt.savefig(OUTPUT_DIR / "per_sentence_accuracy.png")
    plt.close()
    print("  Saved.")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("Generating paper figures...")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print("=" * 60, flush=True)

    fig1_heatmap_same_sentence()
    fig2_heatmap_diff_sentence()
    fig3_similarity_matrix()
    fig5_framewise_curves()
    fig4_score_distribution()
    fig6_topk_accuracy()
    fig7_per_sentence_accuracy()

    print("\n" + "=" * 60)
    print(f"All figures saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
