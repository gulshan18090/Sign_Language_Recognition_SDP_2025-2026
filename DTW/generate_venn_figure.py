"""
Generate a Venn diagram showing Method 1 vs Method 2 success/failure overlap.
Also generates a confusion-style 2x2 contingency table figure.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'latex_template', 'figures')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'recognition_results')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'figure.dpi': 300,
})

def load_json(fname):
    with open(os.path.join(RESULTS_DIR, fname), encoding='utf-8') as f:
        return json.load(f)

# Load both methods
m1_data = load_json('recognition_method6_dtw_cos_cost_summary.json')
m2_data = load_json('recognition_method7_angles_rel_len_summary.json')

# Build per-video top1 correct sets
# Key: (folder, user_video)
m1_correct = set()
m1_wrong = set()
for entry in m1_data['detailed']:
    key = (entry['folder'], entry['user_video'])
    if entry['top1']:
        m1_correct.add(key)
    else:
        m1_wrong.add(key)

m2_correct = set()
m2_wrong = set()
for entry in m2_data['detailed']:
    key = (entry['folder'], entry['user_video'])
    if entry['top1']:
        m2_correct.add(key)
    else:
        m2_wrong.add(key)

# Compute 4 quadrants
both_correct = m1_correct & m2_correct
both_wrong = m1_wrong & m2_wrong
m1_only = m1_correct & m2_wrong     # M1 correct, M2 wrong
m2_only = m2_correct & m1_wrong     # M2 correct, M1 wrong

total = len(both_correct) + len(both_wrong) + len(m1_only) + len(m2_only)

print(f"Both correct:     {len(both_correct)}")
print(f"Both wrong:       {len(both_wrong)}")
print(f"M1-only correct:  {len(m1_only)}")
print(f"M2-only correct:  {len(m2_only)}")
print(f"Total:            {total}")


# ============================================================
# FIGURE: Venn diagram
# ============================================================
def fig_venn():
    fig, ax = plt.subplots(figsize=(4.8, 3.5))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Two overlapping circles
    circle1 = plt.Circle((-0.7, 0), 1.5, fill=False, edgecolor='#2166ac',
                         linewidth=2.5, linestyle='-')
    circle2 = plt.Circle((0.7, 0), 1.5, fill=False, edgecolor='#b2182b',
                         linewidth=2.5, linestyle='-')

    # Fill regions with transparency
    # Both correct (intersection)
    intersection = plt.Circle((0, 0), 0.01, alpha=0)  # placeholder

    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # Fill left only (M1 only correct)
    from matplotlib.patches import Circle
    c1 = Circle((-0.7, 0), 1.5, fc='#2166ac', alpha=0.15)
    c2 = Circle((0.7, 0), 1.5, fc='#b2182b', alpha=0.15)
    ax.add_patch(c1)
    ax.add_patch(c2)

    # Labels inside circles
    # M1-only region (left)
    ax.text(-1.5, 0, f'{len(m1_only)}', ha='center', va='center',
            fontsize=22, fontweight='bold', color='#2166ac')
    ax.text(-1.5, -0.5, 'M1-only\ncorrect', ha='center', va='center',
            fontsize=8, color='#2166ac')

    # M2-only region (right)
    ax.text(1.5, 0, f'{len(m2_only)}', ha='center', va='center',
            fontsize=22, fontweight='bold', color='#b2182b')
    ax.text(1.5, -0.5, 'M2-only\ncorrect', ha='center', va='center',
            fontsize=8, color='#b2182b')

    # Intersection (both correct)
    ax.text(0, 0.1, f'{len(both_correct)}', ha='center', va='center',
            fontsize=22, fontweight='bold', color='#4a1486')
    ax.text(0, -0.4, 'Both\ncorrect', ha='center', va='center',
            fontsize=8, color='#4a1486')

    # Outside (both wrong)
    ax.text(0, -2.1, f'Both wrong: {len(both_wrong)} / {total}',
            ha='center', va='center', fontsize=9, color='#666666',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', edgecolor='grey'))

    # Circle labels
    ax.text(-2.3, 1.7, 'Method 1', ha='center', fontsize=10, fontweight='bold',
            color='#2166ac')
    ax.text(-2.3, 1.3, f'(Top-1: {len(m1_correct)}/215)', ha='center', fontsize=8,
            color='#2166ac')
    ax.text(2.3, 1.7, 'Method 2', ha='center', fontsize=10, fontweight='bold',
            color='#b2182b')
    ax.text(2.3, 1.3, f'(Top-1: {len(m2_correct)}/215)', ha='center', fontsize=8,
            color='#b2182b')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_venn_methods.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'fig_venn_methods.eps'), bbox_inches='tight')
    plt.close()
    print('  [OK] fig_venn_methods')


# ============================================================
# FIGURE: 2x2 contingency matrix (cleaner alternative)
# ============================================================
def fig_contingency():
    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 4)
    ax.axis('off')

    # Grid
    cells = [
        # (x, y, value, color, label)
        (1.5, 2.8, len(both_correct), '#1b7837', 'Both correct'),
        (3.0, 2.8, len(m2_only), '#fc8d59', 'M2-only correct'),
        (1.5, 1.5, len(m1_only), '#67a9cf', 'M1-only correct'),
        (3.0, 1.5, len(both_wrong), '#d73027', 'Both wrong'),
    ]

    for x, y, val, col, label in cells:
        rect = FancyBboxPatch((x-0.6, y-0.5), 1.2, 1.0,
                              boxstyle='round,pad=0.05', facecolor=col, alpha=0.3,
                              edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(x, y + 0.1, str(val), ha='center', va='center',
                fontsize=18, fontweight='bold')
        ax.text(x, y - 0.3, label, ha='center', va='center', fontsize=7)

    # Row/column headers
    ax.text(1.5, 3.6, 'M1 correct', ha='center', fontsize=9, fontweight='bold', color='#2166ac')
    ax.text(3.0, 3.6, 'M1 wrong', ha='center', fontsize=9, fontweight='bold', color='#2166ac')
    ax.text(0.5, 2.8, 'M2\ncorrect', ha='center', va='center', fontsize=9,
            fontweight='bold', color='#b2182b')
    ax.text(0.5, 1.5, 'M2\nwrong', ha='center', va='center', fontsize=9,
            fontweight='bold', color='#b2182b')

    # Totals
    ax.text(1.5, 0.7, f'M1 total: {len(m1_correct)}', ha='center', fontsize=8, color='grey')
    ax.text(3.0, 0.7, f'M2 total: {len(m2_correct)}', ha='center', fontsize=8, color='grey')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig_contingency.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(OUT_DIR, 'fig_contingency.eps'), bbox_inches='tight')
    plt.close()
    print('  [OK] fig_contingency')


if __name__ == '__main__':
    print('Generating Venn/contingency figures...')
    fig_venn()
    fig_contingency()
    print(f'Saved to: {os.path.abspath(OUT_DIR)}')
