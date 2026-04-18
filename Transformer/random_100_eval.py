"""
Randomly sample 100 videos from the dataset, run predictions,
and display statistics comparing predictions vs ground truth.

Usage:
    python random_100_eval.py
    python random_100_eval.py --checkpoint artifacts/checkpoints/best.pt --n 100 --beam_size 4
    python random_100_eval.py --n 50 --seed 123
"""
import argparse
import csv
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from config.config import get_config
from inference.predictor import SLRPredictor
from training.evaluate import evaluate_batch, word_error_rate


def load_csv_records(csv_path: str, sep: str, id_col: str, target_col: str):
    """Load CSV and return list of dicts with idd and gloss."""
    records = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=sep)
        for row in reader:
            if len(row) >= 3:
                records.append({id_col: row[0], target_col: row[2]})
    return records


def filter_existing_videos(records, videos_root, id_col, video_extensions):
    """Keep only records whose video folder exists and contains a video file."""
    valid = []
    for rec in records:
        folder = Path(videos_root) / str(rec[id_col])
        if folder.exists():
            has_video = any(
                list(folder.glob(f"*{ext}")) for ext in video_extensions
            )
            if has_video:
                valid.append(rec)
    return valid


def main():
    parser = argparse.ArgumentParser(description="Random 100 Video Evaluation")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--n", type=int, default=100,
                        help="Number of random videos to evaluate (default: 100)")
    parser.add_argument("--beam_size", type=int, default=4,
                        help="Beam size for decoding (1=greedy, >1=beam search)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--save_csv", type=str, default="random_eval_results.csv",
                        help="Save per-sample results to this CSV file")
    args = parser.parse_args()

    cfg = get_config()

    # ── Load CSV records ─────────────────────────────────────────────────
    print(f"Loading dataset from {cfg.paths.csv_path} ...")
    records = load_csv_records(
        cfg.paths.csv_path, cfg.data.csv_sep,
        cfg.data.id_column, cfg.data.target_column,
    )
    print(f"  Total CSV records: {len(records)}")

    # ── Filter to existing videos ────────────────────────────────────────
    records = filter_existing_videos(
        records, cfg.paths.videos_root,
        cfg.data.id_column, cfg.video.video_extensions,
    )
    print(f"  Valid video records: {len(records)}")

    if len(records) == 0:
        print("ERROR: No valid video records found. Check videos_root path.")
        sys.exit(1)

    # ── Random sample ────────────────────────────────────────────────────
    n = min(args.n, len(records))
    if args.seed is not None:
        random.seed(args.seed)
    sampled = random.sample(records, n)
    print(f"  Sampled {n} videos for evaluation\n")

    # ── Load model ───────────────────────────────────────────────────────
    print("Loading model checkpoint ...")
    predictor = SLRPredictor.from_checkpoint(args.checkpoint, cfg=cfg)
    print()

    # ── Run predictions ──────────────────────────────────────────────────
    predictions = []
    references = []
    per_sample_wer = []
    results_rows = []
    errors = []

    print(f"{'#':>4}  {'IDD':>6}  {'Status':>8}  {'WER':>6}  {'Reference':<35}  {'Prediction':<35}")
    print("─" * 110)

    start_time = time.time()

    for i, rec in enumerate(sampled):
        idd = str(rec[cfg.data.id_column])
        ref_gloss = rec[cfg.data.target_column]
        folder = Path(cfg.paths.videos_root) / idd

        try:
            pred_gloss = predictor.predict_folder(str(folder), beam_size=args.beam_size)

            pred_toks = pred_gloss.lower().split()
            ref_toks = ref_gloss.lower().split()
            sample_wer = word_error_rate(pred_toks, ref_toks)
            exact = pred_toks == ref_toks

            predictions.append(pred_gloss)
            references.append(ref_gloss)
            per_sample_wer.append(sample_wer)

            status = "✓ EXACT" if exact else f"WER={sample_wer:.2f}"
            print(f"{i+1:>4}  {idd:>6}  {status:>8}  {sample_wer:>6.2f}  {ref_gloss:<35}  {pred_gloss:<35}")

            results_rows.append({
                "idd": idd,
                "reference": ref_gloss,
                "prediction": pred_gloss,
                "wer": f"{sample_wer:.4f}",
                "exact_match": "1" if exact else "0",
                "status": "success",
            })

        except Exception as e:
            errors.append((idd, str(e)))
            print(f"{i+1:>4}  {idd:>6}  {'ERROR':>8}  {'N/A':>6}  {ref_gloss:<35}  {str(e)[:35]}")
            results_rows.append({
                "idd": idd,
                "reference": ref_gloss,
                "prediction": "",
                "wer": "",
                "exact_match": "",
                "status": f"error: {str(e)[:80]}",
            })

    elapsed = time.time() - start_time

    # ── Compute aggregate metrics ────────────────────────────────────────
    print("\n" + "═" * 110)
    print("                         EVALUATION STATISTICS")
    print("═" * 110)

    if predictions:
        metrics = evaluate_batch(predictions, references)

        n_evaluated = len(predictions)
        n_exact = sum(1 for w in per_sample_wer if w == 0.0)
        n_partial = sum(1 for w in per_sample_wer if 0.0 < w < 1.0)
        n_wrong = sum(1 for w in per_sample_wer if w >= 1.0)

        # WER distribution buckets
        wer_buckets = {
            "Perfect (WER=0)": 0,
            "Good (WER<0.25)": 0,
            "Fair (0.25≤WER<0.5)": 0,
            "Poor (0.5≤WER<1.0)": 0,
            "Bad (WER≥1.0)": 0,
        }
        for w in per_sample_wer:
            if w == 0.0:
                wer_buckets["Perfect (WER=0)"] += 1
            elif w < 0.25:
                wer_buckets["Good (WER<0.25)"] += 1
            elif w < 0.5:
                wer_buckets["Fair (0.25≤WER<0.5)"] += 1
            elif w < 1.0:
                wer_buckets["Poor (0.5≤WER<1.0)"] += 1
            else:
                wer_buckets["Bad (WER≥1.0)"] += 1

        # Unique glosses in predictions and references
        pred_words = Counter()
        ref_words = Counter()
        for p in predictions:
            pred_words.update(p.lower().split())
        for r in references:
            ref_words.update(r.lower().split())

        print(f"\n  Checkpoint       : {args.checkpoint}")
        print(f"  Beam size        : {args.beam_size}")
        print(f"  Videos sampled   : {n}")
        print(f"  Successfully eval: {n_evaluated}")
        print(f"  Errors           : {len(errors)}")
        print(f"  Time elapsed     : {elapsed:.1f}s  ({elapsed/max(n_evaluated,1):.2f}s per video)")

        print(f"\n  ── Aggregate Metrics ──")
        print(f"  Word Error Rate  : {metrics['wer']:.4f}  (lower is better)")
        print(f"  Sequence Accuracy: {metrics['seq_acc']:.4f}  ({int(metrics['seq_acc']*n_evaluated)}/{n_evaluated} exact matches)")
        print(f"  BLEU-4           : {metrics['bleu4']:.4f}  (higher is better)")

        print(f"\n  ── WER Distribution ──")
        max_bar = 40
        for bucket, count in wer_buckets.items():
            pct = count / n_evaluated * 100
            bar = "█" * int(pct / 100 * max_bar)
            print(f"  {bucket:<25}  {count:>4}  ({pct:5.1f}%)  {bar}")

        print(f"\n  ── WER Statistics ──")
        sorted_wer = sorted(per_sample_wer)
        mean_wer = sum(per_sample_wer) / len(per_sample_wer)
        median_wer = sorted_wer[len(sorted_wer) // 2]
        min_wer = sorted_wer[0]
        max_wer = sorted_wer[-1]
        p25 = sorted_wer[int(len(sorted_wer) * 0.25)]
        p75 = sorted_wer[int(len(sorted_wer) * 0.75)]
        print(f"  Mean   : {mean_wer:.4f}")
        print(f"  Median : {median_wer:.4f}")
        print(f"  Min    : {min_wer:.4f}")
        print(f"  Max    : {max_wer:.4f}")
        print(f"  P25    : {p25:.4f}")
        print(f"  P75    : {p75:.4f}")

        print(f"\n  ── Vocabulary Coverage ──")
        print(f"  Unique words in references  : {len(ref_words)}")
        print(f"  Unique words in predictions : {len(pred_words)}")
        common = set(pred_words.keys()) & set(ref_words.keys())
        only_pred = set(pred_words.keys()) - set(ref_words.keys())
        only_ref = set(ref_words.keys()) - set(pred_words.keys())
        print(f"  Words in both              : {len(common)}")
        print(f"  Words only in predictions  : {len(only_pred)}  {list(only_pred)[:10]}")
        print(f"  Words only in references   : {len(only_ref)}  {list(only_ref)[:10]}")

        # Top predicted words
        print(f"\n  ── Top 15 Predicted Words ──")
        for word, cnt in pred_words.most_common(15):
            print(f"    {word:<20}  {cnt}")

        # Show some best and worst examples
        indexed_wer = list(enumerate(per_sample_wer))
        indexed_wer.sort(key=lambda x: x[1])

        print(f"\n  ── 5 Best Predictions (lowest WER) ──")
        for rank, (idx, w) in enumerate(indexed_wer[:5], 1):
            print(f"    {rank}. [WER={w:.2f}] Ref: \"{references[idx]}\"  →  Pred: \"{predictions[idx]}\"")

        print(f"\n  ── 5 Worst Predictions (highest WER) ──")
        for rank, (idx, w) in enumerate(indexed_wer[-5:], 1):
            print(f"    {rank}. [WER={w:.2f}] Ref: \"{references[idx]}\"  →  Pred: \"{predictions[idx]}\"")

    else:
        print("  No successful predictions were made.")

    if errors:
        print(f"\n  ── Errors ({len(errors)}) ──")
        for idd, err in errors[:10]:
            print(f"    IDD {idd}: {err[:100]}")

    # ── Save CSV ─────────────────────────────────────────────────────────
    if args.save_csv and results_rows:
        csv_path = args.save_csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["idd", "reference", "prediction", "wer", "exact_match", "status"])
            writer.writeheader()
            writer.writerows(results_rows)
        print(f"\n  Results saved to {csv_path}")

    print()


if __name__ == "__main__":
    main()
