"""
Visualize Recognition Accuracy and Rank Distribution
===================================================
Reads recognition_dtw.csv and produces:
- Histogram of correct ranks (Top-1, Top-2, ...)
- Per-video correct/incorrect bar (green/red)
- Prints Top-1, Top-3, Top-5 accuracy
- Combined per-sentence chart (Top-1/3/5 accuracy + average rank)
- Saves PNGs to recognition_results/

Usage:
  python visualize_recognition_accuracy.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

CSV_PATH = os.path.join('recognition_results', 'recognition_dtw.csv')
OUTPUT_DIR = 'recognition_results'

# Read CSV
df = pd.read_csv(CSV_PATH)

# Normalize types for robust plotting
for col in ['top1', 'top3', 'top5']:
    if col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().isin(['true', '1', 'yes'])
        else:
            df[col] = df[col].astype(bool)
df['correct_rank'] = pd.to_numeric(df['correct_rank'], errors='coerce')
df = df.dropna(subset=['correct_rank'])
df['correct_rank'] = df['correct_rank'].astype(int)

# Histogram of correct ranks
plt.figure(figsize=(10,5))
plt.hist(df['correct_rank'], bins=range(1, max(df['correct_rank'])+2), color='#1976D2', edgecolor='black', alpha=0.8)
plt.xlabel('Correct Phrase Rank', fontsize=13)
plt.ylabel('Number of Videos', fontsize=13)
plt.title('Distribution of Correct Phrase Rank (DTW Recognition)', fontsize=15)
plt.xticks(range(1, max(df['correct_rank'])+1))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'correct_rank_histogram.png'), dpi=150)
plt.close()

# Per-video correct/incorrect bar
plt.figure(figsize=(14,3))
colors = df['top1'].map(lambda x: '#43A047' if x else '#E53935')
plt.bar(range(len(df)), [1]*len(df), color=colors, edgecolor='none', width=1.0)
plt.xlabel('Video Index', fontsize=13)
plt.yticks([])
plt.title('Per-Video Top-1 Correctness (Green=Correct, Red=Wrong)', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'per_video_top1_correct.png'), dpi=150)
plt.close()

# Print and plot Top-1, Top-3, Top-5 accuracy
n = len(df)
top1 = df['top1'].sum()
top3 = df['top3'].sum()
top5 = df['top5'].sum()
print(f"Total videos: {n}")
print(f"Top-1 accuracy: {top1}/{n} = {top1/n*100:.2f}%")
print(f"Top-3 accuracy: {top3}/{n} = {top3/n*100:.2f}%")
print(f"Top-5 accuracy: {top5}/{n} = {top5/n*100:.2f}%")

plt.figure(figsize=(7,5))
plt.bar(['Top-1', 'Top-3', 'Top-5'], [top1/n*100, top3/n*100, top5/n*100], color=['#43A047','#1976D2','#FBC02D'])
plt.ylabel('Accuracy (%)', fontsize=13)
plt.ylim(0,100)
plt.title('Recognition Accuracy (DTW)', fontsize=15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'recognition_accuracy_bar.png'), dpi=150)
plt.close()

# Combined per-sentence visualization: accuracy levels + rank
sentence_stats = (
    df.groupby('folder', as_index=False)
    .agg(
        samples=('folder', 'size'),
        top1_acc=('top1', 'mean'),
        top3_acc=('top3', 'mean'),
        top5_acc=('top5', 'mean'),
        avg_rank=('correct_rank', 'mean'),
    )
)
sentence_stats['top1_acc'] = sentence_stats['top1_acc'] * 100.0
sentence_stats['top3_acc'] = sentence_stats['top3_acc'] * 100.0
sentence_stats['top5_acc'] = sentence_stats['top5_acc'] * 100.0
sentence_stats['avg_rank'] = sentence_stats['avg_rank'].round(2)

# Sort to keep stronger sentences first
sentence_stats = sentence_stats.sort_values(
    by=['top1_acc', 'top3_acc', 'top5_acc', 'avg_rank'],
    ascending=[False, False, False, True]
).reset_index(drop=True)

n_sentences = len(sentence_stats)
x = np.arange(n_sentences)
bar_width = 0.24
fig_width = max(18, n_sentences * 0.35)

fig, ax1 = plt.subplots(figsize=(fig_width, 8))

ax1.bar(x - bar_width, sentence_stats['top1_acc'], width=bar_width, label='Top-1 Accuracy', color='#43A047')
ax1.bar(x, sentence_stats['top3_acc'], width=bar_width, label='Top-3 Accuracy', color='#1E88E5')
ax1.bar(x + bar_width, sentence_stats['top5_acc'], width=bar_width, label='Top-5 Accuracy', color='#F9A825')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(x, sentence_stats['avg_rank'], color='#E53935', marker='o', linewidth=2, markersize=4, label='Average Correct Rank')
ax2.set_ylabel('Average Correct Rank (Lower is Better)', fontsize=12)
ax2.set_ylim(0.8, max(sentence_stats['avg_rank'].max() + 2, 5))

ax1.set_xticks(x)
ax1.set_xticklabels(sentence_stats['folder'], rotation=75, ha='right', fontsize=8)
ax1.set_xlabel('Sentence', fontsize=12)
ax1.set_title('Per-Sentence Recognition: Accuracy Levels and Average Rank (DTW)', fontsize=14)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=10)

plt.tight_layout()
combined_path = os.path.join(OUTPUT_DIR, 'sentence_accuracy_rank_combined.png')
plt.savefig(combined_path, dpi=180)
plt.close()

# Save per-sentence metrics table
sentence_csv_path = os.path.join(OUTPUT_DIR, 'per_sentence_accuracy_rank.csv')
sentence_stats.to_csv(sentence_csv_path, index=False)

# Save per-video accuracy/rank as CSV
df[['folder','user_video','correct_rank','top1','top3','top5']].to_csv(os.path.join(OUTPUT_DIR, 'per_video_accuracy.csv'), index=False)
print(
    'Saved: correct_rank_histogram.png, per_video_top1_correct.png, recognition_accuracy_bar.png, '
    'sentence_accuracy_rank_combined.png, per_video_accuracy.csv, per_sentence_accuracy_rank.csv'
)
