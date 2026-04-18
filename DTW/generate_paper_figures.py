"""
Generate publication-quality figures for ITTA 2026 paper.
Outputs EPS + PDF for Springer LNCS compatibility.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import io

# Fix encoding for Azerbaijani text
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'latex_template', 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Springer LNCS: single column ~122mm wide = 4.8in, use ~4.5in for safety
# Font: use serif to match LaTeX body text
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
})

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'recognition_results')

def load_json(fname):
    path = os.path.join(RESULTS_DIR, fname)
    with open(path, encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# FIGURE 1: Grouped bar chart — all 4 methods, Top-1/3/5
# ============================================================
def fig1_method_comparison():
    methods = ['Method 1\n(coords)', 'Method 2\n(angles+len)', 'Method 2w\n(+wrist)', 'Method 2p\n(+palm+$d_1$)']
    top1 = [33.95, 40.00, 40.47, 27.91]
    top3 = [44.65, 50.23, 51.16, 44.19]
    top5 = [50.23, 56.28, 55.81, 52.56]

    x = np.arange(len(methods))
    w = 0.25

    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    bars1 = ax.bar(x - w, top1, w, label='Top-1', color='#2166ac', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x,     top3, w, label='Top-3', color='#67a9cf', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + w, top5, w, label='Top-5', color='#d1e5f0', edgecolor='black', linewidth=0.5)

    # Add chance level line
    ax.axhline(y=1.06, color='red', linestyle='--', linewidth=0.8, label='Chance (1.06%)')

    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 65)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    # Value labels on top of bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.8,
                    f'{h:.1f}', ha='center', va='bottom', fontsize=6.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_method_comparison.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'fig_method_comparison.eps'), bbox_inches='tight')
    plt.close()
    print('  [OK] fig_method_comparison')


# ============================================================
# FIGURE 2: Rank histogram for Method 2
# ============================================================
def fig2_rank_histogram():
    data = load_json('recognition_method7_angles_rel_len_summary.json')
    ranks = []
    for entry in data['detailed']:
        ranks.append(entry['correct_rank'])

    fig, ax = plt.subplots(figsize=(4.8, 2.8))

    max_rank = max(ranks)
    bins = np.arange(1, max_rank + 2) - 0.5
    counts, _, patches = ax.hist(ranks, bins=bins, color='#2166ac', edgecolor='black', linewidth=0.5)

    # Colour Top-1 green, Top-2-5 orange, rest grey
    for i, p in enumerate(patches):
        rank = i + 1
        if rank == 1:
            p.set_facecolor('#1b7837')  # green
        elif rank <= 5:
            p.set_facecolor('#fc8d59')  # orange
        else:
            p.set_facecolor('#bdbdbd')  # grey

    # Add vertical lines for Top-1, Top-5 boundaries
    ax.axvline(x=1.5, color='black', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axvline(x=5.5, color='black', linestyle=':', linewidth=0.8, alpha=0.5)

    # Annotations
    ax.text(1, counts[0] + 2, f'Top-1\n{int(counts[0])}', ha='center', fontsize=7, fontweight='bold')
    top5_count = int(sum(counts[:5]))
    ax.text(3.5, max(counts[1:5]) + 5, f'Top-5: {top5_count}', ha='center', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fc8d59', alpha=0.5))

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#1b7837', edgecolor='black', label='Top-1 correct'),
        mpatches.Patch(facecolor='#fc8d59', edgecolor='black', label='Ranks 2--5'),
        mpatches.Patch(facecolor='#bdbdbd', edgecolor='black', label='Ranks > 5 (far miss)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)

    ax.set_xlabel('Rank of correct phrase')
    ax.set_ylabel('Number of videos')
    ax.set_xlim(0.5, min(max_rank + 0.5, 50))  # Truncate at 50 for readability
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_rank_histogram.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'fig_rank_histogram.eps'), bbox_inches='tight')
    plt.close()
    print('  [OK] fig_rank_histogram')


# ============================================================
# FIGURE 3: Score margin histogram for Method 2 failures
# ============================================================
def fig3_score_margin():
    data = load_json('recognition_method7_angles_rel_len_summary.json')
    margins = []
    for entry in data['detailed']:
        if not entry['top1']:
            # margin = top1_score - correct_score
            margin = entry['top1_score'] - entry['correct_score']
            margins.append(margin)

    margins = np.array(margins)

    fig, ax = plt.subplots(figsize=(4.8, 2.8))

    bins = np.linspace(0, max(margins) + 0.005, 30)
    counts, edges, patches = ax.hist(margins, bins=bins, color='#2166ac', edgecolor='black', linewidth=0.5)

    # Colour tight margins (<0.02) differently
    for i, p in enumerate(patches):
        bin_center = (edges[i] + edges[i+1]) / 2
        if bin_center < 0.02:
            p.set_facecolor('#d73027')  # red = tight
        else:
            p.set_facecolor('#2166ac')  # blue = clear

    # Add vertical lines
    ax.axvline(x=0.02, color='black', linestyle='--', linewidth=0.8)
    ax.text(0.021, max(counts) * 0.9, 'gap = 0.02', fontsize=7, rotation=90, va='top')

    # Stats annotation
    stats_text = (f'n = {len(margins)}\n'
                  f'mean = {np.mean(margins):.3f}\n'
                  f'median = {np.median(margins):.3f}\n'
                  f'tight (<0.02): {np.sum(margins < 0.02)}/{len(margins)}')
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=7,
            va='top', ha='right', bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

    legend_elements = [
        mpatches.Patch(facecolor='#d73027', edgecolor='black', label='Tight margin (<0.02)'),
        mpatches.Patch(facecolor='#2166ac', edgecolor='black', label='Clear margin (>=0.02)'),
    ]
    ax.legend(handles=legend_elements, loc='center right', fontsize=7)

    ax.set_xlabel('Score margin (wrong Top-1 score $-$ correct score)')
    ax.set_ylabel('Number of failures')
    ax.grid(axis='y', alpha=0.3, linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_score_margin.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'fig_score_margin.eps'), bbox_inches='tight')
    plt.close()
    print('  [OK] fig_score_margin')


# ============================================================
# FIGURE 4: System pipeline diagram (matplotlib, not TikZ)
# ============================================================
def fig4_pipeline():
    fig, ax = plt.subplots(figsize=(4.8, 2.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis('off')

    # Box style
    box_kw = dict(boxstyle='round,pad=0.3', facecolor='#d1e5f0', edgecolor='black', linewidth=1)
    box_kw2 = dict(boxstyle='round,pad=0.3', facecolor='#fc8d59', edgecolor='black', linewidth=1)
    box_kw3 = dict(boxstyle='round,pad=0.3', facecolor='#1b7837', edgecolor='black', linewidth=1)

    # Top row: Offline pipeline
    positions = [
        (0.8, 2.2, 'Video\nInput', box_kw),
        (2.6, 2.2, 'MediaPipe\nExtraction', box_kw),
        (4.4, 2.2, 'Preprocess\n(filter+dedup)', box_kw),
        (6.4, 2.2, 'Feature\nExtraction', box_kw),
        (8.2, 2.2, 'DTW\nAlignment', box_kw2),
        (9.5, 2.2, 'Top-K\nRank', box_kw3),
    ]

    for x, y, text, bkw in positions:
        ax.text(x, y, text, ha='center', va='center', fontsize=7,
                bbox=bkw, fontweight='bold' if 'DTW' in text or 'Rank' in text else 'normal',
                color='white' if 'Rank' in text else 'black')

    # Arrows top row
    arrow_kw = dict(arrowstyle='->', color='black', lw=1.2)
    for i in range(len(positions) - 1):
        x1 = positions[i][0] + 0.6
        x2 = positions[i+1][0] - 0.6
        ax.annotate('', xy=(x2, 2.2), xytext=(x1, 2.2), arrowprops=arrow_kw)

    # Bottom row: Feature variants
    ax.text(6.4, 0.8, 'Method 1: 126-D coords\nMethod 2: 420-D angles+len',
            ha='center', va='center', fontsize=6.5,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7f7f7', edgecolor='grey', linewidth=0.8))

    # Arrow from Feature Extraction down to variants
    ax.annotate('', xy=(6.4, 1.15), xytext=(6.4, 1.85),
                arrowprops=dict(arrowstyle='->', color='grey', lw=0.8, linestyle='--'))

    # Streaming label
    ax.text(5.0, 0.3, 'Streaming mode: column-by-column DTW, O(N) per frame',
            ha='center', va='center', fontsize=6.5, style='italic', color='#666666')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_pipeline.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'fig_pipeline.eps'), bbox_inches='tight')
    plt.close()
    print('  [OK] fig_pipeline')


# ============================================================
# FIGURE 5: Streaming state machine
# ============================================================
def fig5_state_machine():
    fig, ax = plt.subplots(figsize=(4.8, 1.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')

    states = [
        (1.2, 1.0, 'WAITING'),
        (3.5, 1.0, 'RECORDING'),
        (6.0, 1.0, 'HAND\nLOST'),
        (8.5, 1.0, 'RESULTS'),
    ]

    colors = ['#d1e5f0', '#67a9cf', '#fc8d59', '#1b7837']

    for (x, y, label), col in zip(states, colors):
        tc = 'white' if col == '#1b7837' else 'black'
        ax.text(x, y, label, ha='center', va='center', fontsize=8, fontweight='bold',
                color=tc,
                bbox=dict(boxstyle='round,pad=0.4', facecolor=col, edgecolor='black', linewidth=1.2))

    # Arrows with labels
    transitions = [
        (1.2, 3.5, 'hand\ndetected', 0.75, 0.7),
        (3.5, 6.0, 'hand\nlost', 0.75, 0.7),
        (6.0, 8.5, '4s\ntimeout', 0.7, 0.65),
    ]

    for x1, x2, label, off1, off2 in transitions:
        ax.annotate('', xy=(x2 - off2, 1.0), xytext=(x1 + off1, 1.0),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
        ax.text((x1 + x2) / 2, 0.35, label, ha='center', va='center', fontsize=6.5,
                color='#333333')

    # Return arrow from HAND_LOST back to RECORDING
    ax.annotate('', xy=(3.5 + 0.75, 1.45), xytext=(6.0 - 0.7, 1.45),
                arrowprops=dict(arrowstyle='->', color='#999999', lw=0.8,
                                connectionstyle='arc3,rad=-0.3'))
    ax.text(4.75, 1.75, 'hand returns\n(<4s)', ha='center', va='center', fontsize=6, color='#999999')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_state_machine.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'fig_state_machine.eps'), bbox_inches='tight')
    plt.close()
    print('  [OK] fig_state_machine')


# ============================================================
# Run all
# ============================================================
if __name__ == '__main__':
    print('Generating paper figures...')
    fig1_method_comparison()
    fig2_rank_histogram()
    fig3_score_margin()
    fig4_pipeline()
    fig5_state_machine()
    print(f'\nAll figures saved to: {os.path.abspath(OUT_DIR)}')
