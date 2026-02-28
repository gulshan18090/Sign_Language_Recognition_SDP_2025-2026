"""
Build dataset CSV from the Videos/ folder structure.

The Videos/ folder has sentence-named folders, each containing video files.
This script scans the folder structure and creates a CSV mapping:
    sentence_id ; sentence_text ; sign_language_gloss

Since the Videos/ folder may contain sentences NOT in the original
sentences_all.csv, we handle both matched and unmatched sentences.

Usage:
    python build_dataset.py                 # Build CSV
    python build_dataset.py --stats         # Show statistics only
"""
import os
import sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import argparse
import pandas as pd

from feature_extraction_telegram.config_telegram import VIDEO_DIR, CSV_PATH


def build_telegram_csv(video_dir, original_csv=None, output_path=None):
    """
    Build sentences CSV from Videos/ folder structure.

    Each subfolder name is the sentence text.
    We try to match with the original sentences_all.csv for sign language glosses.
    For unmatched sentences, we use the sentence text as-is for the gloss.
    """
    # Load original CSV for sign language gloss mapping
    original_mapping = {}
    if original_csv and os.path.exists(original_csv):
        orig_df = pd.read_csv(
            original_csv, sep=';', encoding='utf-8',
            header=None, names=['idd', 'sentence', 'sign_language']
        )
        for _, row in orig_df.iterrows():
            original_mapping[str(row['sentence']).strip()] = str(row['sign_language']).strip()

    rows = []
    sentence_id = 1

    for folder_name in sorted(os.listdir(video_dir)):
        folder_path = os.path.join(video_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Count videos in this folder
        videos = [f for f in os.listdir(folder_path)
                  if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not videos:
            continue

        sentence_text = folder_name.strip()

        # Try to find sign language gloss from original CSV
        sign_gloss = original_mapping.get(sentence_text, None)

        if sign_gloss is None:
            # Use the sentence text itself as gloss (lowercase, basic tokenization)
            sign_gloss = sentence_text.lower()

        rows.append({
            'idd': sentence_id,
            'sentence': sentence_text,
            'sign_language': sign_gloss,
            'num_videos': len(videos),
            'folder_path': folder_path
        })
        sentence_id += 1

    df = pd.DataFrame(rows)

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Save in same format as sentences_all.csv (semicolon-separated, no header)
        csv_df = df[['idd', 'sentence', 'sign_language']]
        csv_df.to_csv(output_path, sep=';', index=False, header=False, encoding='utf-8')
        print(f"✅ Saved telegram CSV: {output_path}")
        print(f"   Total sentences: {len(df)}")
        print(f"   Total videos: {df['num_videos'].sum()}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Build Telegram Dataset CSV")
    parser.add_argument("--stats", action="store_true", help="Show statistics only")
    args = parser.parse_args()

    original_csv = os.path.join(PARENT_DIR, "drive", "sentences_all.csv")

    df = build_telegram_csv(
        VIDEO_DIR,
        original_csv=original_csv,
        output_path=None if args.stats else CSV_PATH
    )

    if df.empty:
        print(f"❌ No videos found in {VIDEO_DIR}")
        return

    print(f"\n{'='*60}")
    print(f"Telegram Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Sentence folders: {len(df)}")
    print(f"  Total videos: {df['num_videos'].sum()}")
    print(f"  Videos per sentence: min={df['num_videos'].min()}, "
          f"max={df['num_videos'].max()}, "
          f"avg={df['num_videos'].mean():.1f}")

    # Show matched vs unmatched with original CSV
    if original_csv and os.path.exists(original_csv):
        orig_df = pd.read_csv(
            original_csv, sep=';', encoding='utf-8',
            header=None, names=['idd', 'sentence', 'sign_language']
        )
        orig_sentences = set(orig_df['sentence'].astype(str).str.strip())
        matched = df[df['sentence'].isin(orig_sentences)]
        unmatched = df[~df['sentence'].isin(orig_sentences)]
        print(f"\n  Matched with sentences_all.csv: {len(matched)}")
        print(f"  New sentences (not in CSV): {len(unmatched)}")

    print(f"\nSample entries:")
    for _, row in df.head(10).iterrows():
        print(f"  [{row['idd']}] {row['sentence'][:50]} ({row['num_videos']} videos)")


if __name__ == "__main__":
    main()
