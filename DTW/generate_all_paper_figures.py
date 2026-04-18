"""
Generate ALL publication-quality figures for ITTA 2026 paper.
Unified minimalist style: grayscale-friendly, consistent fonts, professional.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
import os, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'latex_template', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'recognition_results')

# ============================================================
# UNIFIED STYLE
# ============================================================
C_DARK   = '#2c3e50'
C_MID    = '#7f8c8d'
C_LIGHT  = '#bdc3c7'
C_GREEN  = '#27ae60'
C_BLUE   = '#2980b9'
C_RED    = '#c0392b'
C_ORANGE = '#e67e22'
C_PURPLE = '#8e44ad'
C_TEAL   = '#16a085'

# Similarity colormap: 0 -> red, mid -> orange/yellow, 1 -> green
SIM_CMAP = LinearSegmentedColormap.from_list(
    'sim_red_yellow_green',
    ['#c0392b', '#e67e22', '#f1c40f', '#27ae60']
)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'grid.alpha': 1.0,
    'grid.linewidth': 0.5,
})

def load_json(fname):
    with open(os.path.join(RESULTS_DIR, fname), encoding='utf-8') as f:
        return json.load(f)

def savefig(name):
    for ext in ['pdf', 'eps']:
        plt.savefig(os.path.join(OUT_DIR, f'{name}.{ext}'), bbox_inches='tight')
    plt.close()
    print(f'  [OK] {name}')


# ============================================================
# FIG 1: System pipeline — dual-track showing both methods
# ============================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(8.0, 4.1))
    ax.set_xlim(0.0, 16.0)
    ax.set_ylim(0.0, 5.2)
    ax.axis('off')

    def step_box(x, y, w, h, text, face, edge=C_DARK, text_color=C_DARK,
                 fontsize=8, bold=False):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle='round,pad=0.03,rounding_size=0.08',
            linewidth=0.9, edgecolor=edge, facecolor=face
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2, y + h / 2, text,
            ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight='bold' if bold else 'normal'
        )

    def flow(x1, y1, x2, y2, color, lw=1.2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle='->', mutation_scale=9,
            linewidth=lw, color=color
        ))

    # Title
    ax.text(8.0, 4.95, 'Two Parallel Recognition Pipelines', ha='center',
            va='center', fontsize=11, fontweight='bold', color=C_DARK)

    # Soft lane backgrounds to separate methods visually.
    ax.add_patch(FancyBboxPatch((4.55, 2.7), 8.0, 1.3,
                                boxstyle='round,pad=0.02,rounding_size=0.07',
                                linewidth=0.0, facecolor='#edf5fb'))
    ax.add_patch(FancyBboxPatch((4.55, 1.0), 8.0, 1.3,
                                boxstyle='round,pad=0.02,rounding_size=0.07',
                                linewidth=0.0, facecolor='#fdf0ee'))

    # Shared preprocessing
    sx, sy, sw, sh = 0.45, 1.90, 3.25, 1.55
    step_box(sx, sy, sw, sh,
             'Shared Preprocessing\nVideo -> MediaPipe ->\nHand Filter -> Dedup',
             face='#e6eef5', fontsize=7.3, bold=True)

    # Lane labels
    ax.text(4.65, 4.15, 'Method 1 (coordinate-based)', ha='left',
            va='center', fontsize=8.5, fontweight='bold', color=C_BLUE)
    ax.text(4.65, 2.45, 'Method 2 (geometry-invariant)', ha='left',
            va='center', fontsize=8.5, fontweight='bold', color=C_RED)

    # Consistent box geometry per lane
    bw, bh, gap = 1.85, 0.72, 0.50
    x0 = 4.85
    y1 = 3.15
    y2 = 1.45

    m1_steps = ['Wrist-center\n+ unit norm', '126-D\ncoordinates',
                'Cosine DTW\nw = 0.30', 'Cosine\nsimilarity']
    m2_steps = ['Bone vectors\n(20/hand)', '420-D angles\n+ rel. lengths',
                'L1 DTW\nw = 0.25', 'L1-based\nsimilarity']

    for i, t in enumerate(m1_steps):
        step_box(x0 + i * (bw + gap), y1, bw, bh, t, face='#dbe9f4')
    for i, t in enumerate(m2_steps):
        step_box(x0 + i * (bw + gap), y2, bw, bh, t, face='#f4dbd8')

    # Branch arrows from shared block.
    flow(sx + sw, sy + sh * 0.72, x0 - 0.18, y1 + bh / 2, C_BLUE)
    flow(sx + sw, sy + sh * 0.28, x0 - 0.18, y2 + bh / 2, C_RED)

    # Horizontal arrows within lanes.
    for i in range(3):
        xa = x0 + i * (bw + gap) + bw
        xb = x0 + (i + 1) * (bw + gap)
        flow(xa + 0.05, y1 + bh / 2, xb - 0.05, y1 + bh / 2, C_BLUE)
        flow(xa + 0.05, y2 + bh / 2, xb - 0.05, y2 + bh / 2, C_RED)

    # Final ranking node.
    rx, ry, rw, rh = 13.95, 2.05, 1.9, 1.1
    step_box(rx, ry, rw, rh, 'Top-K\nranking\n(94 classes)',
             face='#d9eee0', fontsize=8.4, bold=True)
    flow(x0 + 3 * (bw + gap) + bw + 0.05, y1 + bh / 2, rx - 0.1, ry + rh * 0.68, C_BLUE)
    flow(x0 + 3 * (bw + gap) + bw + 0.05, y2 + bh / 2, rx - 0.1, ry + rh * 0.32, C_RED)

    savefig('fig_pipeline')


# ============================================================
# FIG 2: State machine
# ============================================================
def fig_state_machine():
    fig, ax = plt.subplots(figsize=(5.0, 1.8))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(-0.2, 2.8)
    ax.axis('off')

    states = [
        (1.2, 1.3, 'WAITING',    '#f0f0f0', '(idle)'),
        (3.8, 1.3, 'RECORDING',  '#d5e8f0', '(signing)'),
        (6.4, 1.3, 'HAND\nLOST', '#fde8d0', '(timeout\ncounter)'),
        (9.0, 1.3, 'RESULTS',    '#d4edda', '(ranked\nscores)'),
    ]

    for x, y, label, col, sub in states:
        ax.text(x, y, label, ha='center', va='center', fontsize=8,
                fontweight='bold', color=C_DARK,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=col,
                          edgecolor=C_DARK, linewidth=1.0))
        ax.text(x, y - 0.7, sub, ha='center', va='center', fontsize=6,
                color=C_MID, style='italic')

    transitions = [
        (1.2, 3.8, 'hand detected', 0.85, 0.85),
        (3.8, 6.4, 'hand lost',     0.85, 0.85),
        (6.4, 9.0, '4 s timeout',   0.80, 0.80),
    ]
    for x1, x2, label, off1, off2 in transitions:
        ax.annotate('', xy=(x2 - off2, 1.3), xytext=(x1 + off1, 1.3),
                    arrowprops=dict(arrowstyle='->', color=C_DARK, lw=1.0))
        ax.text((x1 + x2) / 2, 1.85, label, ha='center', va='center',
                fontsize=6.5, color=C_MID,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          edgecolor=C_LIGHT, linewidth=0.4))

    # Return arrow
    ax.annotate('', xy=(3.8 + 0.85, 2.2), xytext=(6.4 - 0.85, 2.2),
                arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=0.9,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(5.1, 2.65, 'hand returns\n(< 4 s)', ha='center', fontsize=6,
            color=C_GREEN)

    savefig('fig_state_machine')


# ============================================================
# FIG 3: Grouped bar chart — all 4 methods
# ============================================================
def fig_method_comparison():
    methods = ['Method 1\n(coordinates)', 'Method 2\n(angles + rel. len.)',
               'Method 2w\n(+ wrist pos.)', 'Method 2p\n(+ palm + $d_1$)']
    top1 = [33.95, 40.00, 40.47, 27.91]
    top3 = [44.65, 50.23, 51.16, 44.19]
    top5 = [50.23, 56.28, 55.81, 52.56]

    x = np.arange(len(methods))
    w = 0.22

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    bars1 = ax.bar(x - w, top1, w, label='Top-1', color=C_DARK,
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x,     top3, w, label='Top-3', color=C_BLUE,
                   edgecolor='white', linewidth=0.5)
    bars3 = ax.bar(x + w, top5, w, label='Top-5', color=C_LIGHT,
                   edgecolor=C_MID, linewidth=0.5)

    # Chance level
    ax.axhline(y=1.06, color=C_RED, linestyle='--', linewidth=0.8)
    ax.text(3.5, 3.5, 'Chance = 1/94 = 1.06%', fontsize=6.5, color=C_RED,
            ha='right')

    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=7)
    ax.set_ylim(0, 66)
    ax.legend(loc='upper right', framealpha=1.0, edgecolor=C_LIGHT,
              ncol=3, fontsize=7)
    ax.grid(axis='y')

    # Value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.8,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=5.5,
                    color=C_DARK)

    savefig('fig_method_comparison')


# ============================================================
# FIG 4: Venn diagram — Top-1 correctness overlap
# ============================================================
def fig_venn():
    m1_data = load_json('recognition_method6_dtw_cos_cost_summary.json')
    m2_data = load_json('recognition_method7_angles_rel_len_summary.json')

    all_keys = set()
    m1_correct = set()
    m2_correct = set()

    for e in m1_data['detailed']:
        key = (e['folder'], e['user_video'])
        all_keys.add(key)
        if e['top1']:
            m1_correct.add(key)

    for e in m2_data['detailed']:
        key = (e['folder'], e['user_video'])
        if e['top1']:
            m2_correct.add(key)

    N = len(all_keys)
    both_c  = len(m1_correct & m2_correct)
    m1_only = len(m1_correct - m2_correct)
    m2_only = len(m2_correct - m1_correct)
    both_w  = N - both_c - m1_only - m2_only

    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    ax.set_xlim(-3.8, 3.8)
    ax.set_ylim(-3.2, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.text(0, 2.55, 'Top-1 Correctness Overlap (N=215)', ha='center',
            fontsize=9, fontweight='bold', color=C_DARK)

    y0 = 0.15
    c1 = Circle((-0.75, y0), 1.65, fc='#eaf2fb', ec=C_BLUE, lw=2.0)
    c2 = Circle((0.75, y0), 1.65, fc='#fbeeee', ec=C_RED, lw=2.0)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # Method-1-only Top-1 successes.
    ax.text(-1.7, y0 + 0.2, f'{m1_only}', ha='center', va='center',
            fontsize=18, fontweight='bold', color=C_BLUE)
    ax.text(-1.7, y0 - 0.3, f'{m1_only/N*100:.1f}%', ha='center',
            fontsize=6.8, color=C_BLUE)

    # Method-2-only Top-1 successes.
    ax.text(1.7, y0 + 0.2, f'{m2_only}', ha='center', va='center',
            fontsize=18, fontweight='bold', color=C_RED)
    ax.text(1.7, y0 - 0.3, f'{m2_only/N*100:.1f}%', ha='center',
            fontsize=6.8, color=C_RED)

    # Shared Top-1 successes.
    ax.text(0, y0 + 0.2, f'{both_c}', ha='center', va='center',
            fontsize=19, fontweight='bold', color=C_DARK)
    ax.text(0, y0 - 0.3, f'{both_c/N*100:.1f}%', ha='center',
            fontsize=6.8, color=C_DARK)

    # Shared failures.
    ax.text(0, -2.45,
            f'Both wrong (Top-1 failure in both): {both_w} ({both_w/N*100:.1f}%)',
            ha='center', fontsize=7.5, color=C_MID,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f5f5f5',
                      edgecolor=C_LIGHT, linewidth=0.6))

    # Top summary labels in separate callout boxes to avoid mixing with circles.
    ax.text(-1.85, 2.20, f'M1 success\n{len(m1_correct)}/{N} ({len(m1_correct)/N*100:.1f}%)',
            fontsize=7.0, color=C_BLUE, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.20', facecolor='white',
                      edgecolor=C_BLUE, linewidth=0.8))
    ax.text(1.85, 2.20, f'M2 success\n{len(m2_correct)}/{N} ({len(m2_correct)/N*100:.1f}%)',
            fontsize=7.0, color=C_RED, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.20', facecolor='white',
                      edgecolor=C_RED, linewidth=0.8))

    savefig('fig_venn_methods')


# ============================================================
# FIG 5: Rank histogram (Method 2)
# ============================================================
def fig_rank_histogram():
    d1 = load_json('recognition_method6_dtw_cos_cost_summary.json')
    d2 = load_json('recognition_method7_angles_rel_len_summary.json')
    r1 = [e['correct_rank'] for e in d1['detailed']]
    r2 = [e['correct_rank'] for e in d2['detailed']]

    max_rank = int(max(max(r1), max(r2)))
    x_max = min(max_rank, 50)
    bins = np.arange(1, x_max + 2) - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.0), sharey=True)

    def draw_panel(ax, ranks, title):
        counts, _, patches = ax.hist(ranks, bins=bins, color=C_LIGHT,
                                     edgecolor='white', linewidth=0.5)
        for i, p in enumerate(patches):
            rank = i + 1
            if rank == 1:
                p.set_facecolor(C_GREEN)
            elif rank <= 5:
                p.set_facecolor(C_ORANGE)
            else:
                p.set_facecolor(C_LIGHT)

        ax.axvline(x=1.5, color=C_DARK, linestyle=':', linewidth=0.6)
        ax.axvline(x=5.5, color=C_DARK, linestyle=':', linewidth=0.6)

        top1_n = int(counts[0]) if len(counts) > 0 else 0
        top5_n = int(sum(counts[:5]))
        near_n = top5_n - top1_n
        far_n = len(ranks) - top5_n
        n = len(ranks)

        if len(counts) > 0:
            ax.text(1, counts[0] + max(1.5, 0.02 * n), f'{top1_n}',
                    ha='center', fontsize=8, fontweight='bold', color=C_GREEN)

        leg = [
            mpatches.Patch(facecolor=C_GREEN, edgecolor='white',
                           label=f'Top-1: {top1_n} ({top1_n/n*100:.1f}%)'),
            mpatches.Patch(facecolor=C_ORANGE, edgecolor='white',
                           label=f'Ranks 2-5: {near_n} ({near_n/n*100:.1f}%)'),
            mpatches.Patch(facecolor=C_LIGHT, edgecolor=C_MID,
                           label=f'Ranks >5: {far_n} ({far_n/n*100:.1f}%)'),
        ]
        ax.legend(handles=leg, loc='upper right', framealpha=1.0,
                  edgecolor=C_LIGHT, fontsize=6.5)
        ax.set_title(title, fontsize=8.6, color=C_DARK)
        ax.set_xlim(0.5, x_max)
        ax.grid(axis='y')

    draw_panel(axes[0], r1, 'Method 1')
    draw_panel(axes[1], r2, 'Method 2')

    axes[0].set_ylabel('Number of test videos')
    axes[0].set_xlabel('Rank of correct phrase')
    axes[1].set_xlabel('Rank of correct phrase')

    fig.suptitle('Rank Distribution of Correct Phrase (N=215)', fontsize=9.2, color=C_DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    savefig('fig_rank_histogram')


# ============================================================
# FIG 6: Score margin histogram (Method 1 vs Method 2)
# ============================================================
def fig_score_margin():
    d1 = load_json('recognition_method6_dtw_cos_cost_summary.json')
    d2 = load_json('recognition_method7_angles_rel_len_summary.json')

    m1 = np.array([e['top1_score'] - e['correct_score'] for e in d1['detailed'] if not e['top1']], dtype=np.float32)
    m2 = np.array([e['top1_score'] - e['correct_score'] for e in d2['detailed'] if not e['top1']], dtype=np.float32)

    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0), sharey=True)

    def draw_panel(ax, margins, title):
        bins = np.linspace(0, float(np.max(margins)) + 0.005, 30)
        counts, edges, patches = ax.hist(margins, bins=bins, color=C_BLUE,
                                         edgecolor='white', linewidth=0.5)

        for i, p in enumerate(patches):
            mid = (edges[i] + edges[i + 1]) / 2
            if mid < 0.02:
                p.set_facecolor(C_RED)
            elif mid < 0.05:
                p.set_facecolor(C_ORANGE)
            else:
                p.set_facecolor(C_BLUE)

        ax.axvline(x=0.02, color=C_DARK, linestyle='--', linewidth=0.8)
        ax.text(0.0225, max(counts) * 0.92, '0.02', fontsize=6.5,
                rotation=90, va='top', color=C_DARK)

        tight = int(np.sum(margins < 0.02))
        mid_count = int(np.sum((margins >= 0.02) & (margins < 0.05)))
        clear = int(np.sum(margins >= 0.05))
        n = len(margins)

        stats = (f'n = {n} failures\n'
                 f'mean = {np.mean(margins):.3f}\n'
                 f'median = {np.median(margins):.3f}')
        ax.text(0.97, 0.95, stats, transform=ax.transAxes, fontsize=6.5,
                va='top', ha='right',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8f8f8',
                          edgecolor=C_LIGHT, linewidth=0.6))

        legend_elements = [
            mpatches.Patch(facecolor=C_RED, edgecolor='white',
                           label=f'<0.02: {tight} ({tight/n*100:.1f}%)'),
            mpatches.Patch(facecolor=C_ORANGE, edgecolor='white',
                           label=f'0.02-0.05: {mid_count}'),
            mpatches.Patch(facecolor=C_BLUE, edgecolor='white',
                           label=f'>=0.05: {clear}'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', framealpha=1.0,
                  edgecolor=C_LIGHT, fontsize=6.3)
        ax.set_title(title, fontsize=8.6, color=C_DARK)
        ax.set_xlabel('Score margin (wrong Top-1 - correct)')
        ax.grid(axis='y')

    draw_panel(axes[0], m1, 'Method 1')
    draw_panel(axes[1], m2, 'Method 2')
    axes[0].set_ylabel('Number of failure cases')
    fig.suptitle('Failure Margin Distribution by Method', fontsize=9.2, color=C_DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    savefig('fig_score_margin')


# ============================================================
# FIG 7: DTW warping path — synthetic realistic example
# ============================================================
def fig_dtw_warping_path():
    """
    Build warping-path comparison from ACTUAL computations on one real
    translator-user pair (not screenshot cropping).
    """
    try:
        from compare_three_methods import extract_all_frames_and_filter
        from method6_dtw_cos_cost import dtw_align_cosine_cost
        from method7_angles_rel_len import features_from_coords_matrix, dtw_global_align
    except Exception as e:
        print(f'  [WARN] Could not import DTW modules for fig_dtw_warping_path: {e}')
        return

    base = Path(__file__).resolve().parent
    videos_dir = base / 'Videos'
    m6 = load_json('recognition_method6_dtw_cos_cost_summary.json')['detailed']
    m7 = load_json('recognition_method7_angles_rel_len_summary.json')['detailed']
    m6_map = {(e['folder'], e['user_video']): e for e in m6}
    m7_map = {(e['folder'], e['user_video']): e for e in m7}

    # Use a pair that is Top-1 correct in BOTH methods.
    # Prefer the requested phrase if available.
    demo_key = None
    shared_keys = sorted(set(m6_map.keys()) & set(m7_map.keys()))
    both_correct = [k for k in shared_keys if bool(m6_map[k]['top1']) and bool(m7_map[k]['top1'])]
    preferred_folder = '21 Mart bayram günüdür'
    preferred = [k for k in both_correct if k[0] == preferred_folder]
    if preferred:
        demo_key = preferred[0]
    elif both_correct:
        demo_key = both_correct[0]
    elif shared_keys:
        demo_key = shared_keys[0]
    else:
        print('  [WARN] No overlapping keys for fig_dtw_warping_path')
        return

    folder, user_video_name = demo_key
    folder_dir = videos_dir / folder
    trans_list = sorted(folder_dir.glob('translator_*.mp4'))
    user_path = folder_dir / user_video_name
    if not trans_list or not user_path.exists():
        print('  [WARN] Could not find demo videos for fig_dtw_warping_path')
        return
    trans_path = trans_list[0]

    ref_coords, ref_idx = extract_all_frames_and_filter(str(trans_path), similarity_threshold=0.99)
    usr_coords, usr_idx = extract_all_frames_and_filter(str(user_path), similarity_threshold=0.99)
    if ref_coords.size == 0 or usr_coords.size == 0:
        print('  [WARN] Empty features for fig_dtw_warping_path')
        return

    r6 = dtw_align_cosine_cost(ref_coords, usr_coords, ref_idx, usr_idx, window_ratio=0.30)
    ref_m7 = features_from_coords_matrix(ref_coords.astype(np.float32, copy=False))
    usr_m7 = features_from_coords_matrix(usr_coords.astype(np.float32, copy=False))
    r7 = dtw_global_align(ref_m7, usr_m7, ref_idx, usr_idx, window_ratio=0.25, swap_mode='global')

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.0))
    items = [
        (axes[0], r6.path, 'Method 1 (Cosine DTW)', C_BLUE, float(r6.score), bool(m6_map[demo_key]['top1'])),
        (axes[1], r7.path, 'Method 2 (L1 DTW)', C_RED, float(r7.score), bool(m7_map[demo_key]['top1'])),
    ]

    for ax, path, ttl, color, score, is_top1 in items:
        if path:
            px = np.array([p[1] for p in path], dtype=np.float32)  # user frame index
            py = np.array([p[0] for p in path], dtype=np.float32)  # reference frame index
            ps = np.array([p[2] for p in path], dtype=np.float32)  # local similarity

            if len(path) >= 2:
                pts = np.column_stack([px, py]).reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                seg_sim = 0.5 * (ps[:-1] + ps[1:])
                lc = LineCollection(segs, cmap=SIM_CMAP, norm=Normalize(0.0, 1.0))
                lc.set_array(seg_sim)
                lc.set_linewidth(1.6)
                ax.add_collection(lc)
                cb = fig.colorbar(lc, ax=ax, fraction=0.045, pad=0.02)
                if 'Cosine' in ttl:
                    cb.set_label('local cosine sim', fontsize=6.0)
                else:
                    cb.set_label('local L1-based sim', fontsize=6.0)
                cb.ax.tick_params(labelsize=6)
            else:
                ax.plot(px, py, color=color, linewidth=1.4)

            ax.scatter([px[0], px[-1]], [py[0], py[-1]],
                       c=[C_GREEN, C_ORANGE], s=14, zorder=3, edgecolors='none')
            dmax = min(max(px), max(py))
            ax.plot([0, dmax], [0, dmax], linestyle='--',
                    color=C_MID, linewidth=0.7)
        ax.set_title(ttl, fontsize=8.4, color=C_DARK)
        ax.set_xlabel('User frame index')
        ax.set_ylabel('Reference frame index')
        status = 'Top-1 correct' if is_top1 else 'Top-1 wrong'
        ax.text(0.03, 0.97, f'score={score:.3f}\n{status}',
                transform=ax.transAxes, va='top', ha='left', fontsize=6.4,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=C_LIGHT, linewidth=0.5))
        ax.grid(linewidth=0.4)

    folder_short = folder if len(folder) < 30 else folder[:27] + '...'
    fig.suptitle(
        f'Actual warping paths on one real pair ({folder_short})',
        fontsize=8.6, color=C_DARK, y=1.02
    )
    plt.tight_layout()
    savefig('fig_dtw_warping_path')


# ============================================================
# FIG 8: Method comparison contingency (detailed breakdown)
# ============================================================
def fig_contingency():
    """Detailed contingency table as a visual heatmap."""
    m1_data = load_json('recognition_method6_dtw_cos_cost_summary.json')
    m2_data = load_json('recognition_method7_angles_rel_len_summary.json')

    # Build contingency for Top-1
    m1_map = {}
    for e in m1_data['detailed']:
        key = (e['folder'], e['user_video'])
        m1_map[key] = e['top1']

    m2_map = {}
    for e in m2_data['detailed']:
        key = (e['folder'], e['user_video'])
        m2_map[key] = e['top1']

    all_keys = sorted(set(m1_map.keys()) & set(m2_map.keys()))

    # 2x2 contingency
    labels = ['Correct', 'Wrong']
    matrix = np.zeros((2, 2), dtype=int)
    for key in all_keys:
        r = 0 if m1_map[key] else 1
        c = 0 if m2_map[key] else 1
        matrix[r, c] += 1

    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    colors = np.array([
        [C_GREEN,  C_ORANGE],
        [C_BLUE,   C_RED],
    ])

    for i in range(2):
        for j in range(2):
            color = colors[i, j]
            y = 1 - i
            rect = plt.Rectangle((j - 0.5, y - 0.5), 1, 1, linewidth=2,
                                 edgecolor='white', facecolor=color)
            ax.add_patch(rect)
            ax.text(j, y + 0.05, str(matrix[i, j]),
                    ha='center', va='center', fontsize=22,
                    fontweight='bold', color='white')
            pct = matrix[i, j] / len(all_keys) * 100
            ax.text(j, y - 0.25, f'({pct:.1f}%)',
                    ha='center', va='center', fontsize=8, color='white')

    # Category labels in cells
    cell_labels = [
        (0, 0, 'Both\ncorrect'),
        (0, 1, 'M1 only\ncorrect'),
        (1, 0, 'M2 only\ncorrect'),
        (1, 1, 'Both\nwrong'),
    ]
    for i, j, lbl in cell_labels:
        y = 1 - i
        ax.text(j, y + 0.32, lbl, ha='center', va='center',
                fontsize=6.8, color='black', style='italic',
                bbox=dict(boxstyle='round,pad=0.12', facecolor='white',
                          edgecolor='none'))

    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.9, 1.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['M2 Correct', 'M2 Wrong'], fontsize=9, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['M1 Wrong', 'M1 Correct'], fontsize=9, fontweight='bold')
    ax.set_xlabel('Method 2 (Top-1)', fontsize=10)
    ax.set_ylabel('Method 1 (Top-1)', fontsize=10)

    # Remove spines for this plot
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)

    # McNemar annotation
    ax.text(0.5, -0.72,
            f'McNemar discordant pairs: {matrix[0,1]} + {matrix[1,0]} = {matrix[0,1]+matrix[1,0]}',
            ha='center', fontsize=8, color=C_MID,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='#f8f8f8',
                      edgecolor=C_LIGHT, linewidth=0.5))

    savefig('fig_contingency')


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    print('Generating all paper figures (unified style)...')
    fig_pipeline()
    fig_state_machine()
    fig_method_comparison()
    fig_venn()
    fig_rank_histogram()
    fig_score_margin()
    fig_dtw_warping_path()
    fig_contingency()
    print(f'\nAll 8 figures saved to: {os.path.abspath(OUT_DIR)}')
