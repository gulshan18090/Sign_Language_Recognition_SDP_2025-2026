"""
Unified research evaluation for Seq2Seq SLR model.

Outputs:
- Top-1 / Top-3 / Top-5 accuracy
- Detailed per-video predictions
- Per-class metrics
- Confusion summary
- Latency breakdown
"""
import os
import time
import json
import argparse
from collections import defaultdict

import pandas as pd
import torch

from config import Config as config
from data.vocab import build_word_dict
from data.dataset_builder import load_sentences
from inference import (
    load_models,
    get_video_features,
    preprocess_features,
    decode_sequence_beam,
)


def normalize_text(s):
    if s is None:
        return ""
    return " ".join(str(s).lower().strip().split())


def load_label_map(csv_path):
    df = pd.read_csv(csv_path, sep=';', header=None, names=['id', 'sentence', 'sign_language'])
    return {str(row['id']): str(row['sign_language']) for _, row in df.iterrows()}


def collect_videos(video_root):
    videos = []
    for root, _, files in os.walk(video_root):
        for file in files:
            if file.endswith('.mp4') or file.endswith('.avi'):
                videos.append(os.path.join(root, file))
    return sorted(videos)


def save_confusion_tables(results_df, out_dir, top_n=25):
    if results_df.empty:
        return

    misclassified = results_df[results_df["top1_match"] == False]
    confusion_counts = (
        misclassified.groupby(["expected", "pred_top1"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    confusion_counts.to_csv(os.path.join(out_dir, "confusion_pairs.csv"), index=False)

    top_expected = (
        results_df["expected"].value_counts().head(top_n).index.tolist()
    )
    top_df = results_df[results_df["expected"].isin(top_expected)].copy()
    matrix = pd.crosstab(top_df["expected"], top_df["pred_top1"], dropna=False)
    matrix.to_csv(os.path.join(out_dir, "confusion_matrix_top_classes.csv"))


def main():
    parser = argparse.ArgumentParser(description="Research evaluation for Seq2Seq SLR")
    parser.add_argument("--csv-path", default="drive/sentences_all.csv", help="Ground-truth CSV path")
    parser.add_argument("--video-dir", default="drive/Video/Cam2", help="Video root directory")
    parser.add_argument("--features-dir", default="features", help="Feature cache directory")
    parser.add_argument("--output-dir", default="research_outputs/seq2seq_eval", help="Output directory")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width for decoding")
    parser.add_argument("--top-k", type=int, default=5, help="Max hypotheses used for Top-K metrics")
    parser.add_argument("--max-videos", type=int, default=0, help="Limit number of videos (0 = all)")
    parser.add_argument("--require-feature-cache", action="store_true",
                        help="Skip videos missing cached feature files (no on-the-fly extraction)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    label_map = load_label_map(args.csv_path)
    videos = collect_videos(args.video_dir)
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    print("=" * 80)
    print("SEQ2SEQ RESEARCH EVALUATION")
    print("=" * 80)
    print(f"Videos discovered: {len(videos)}")
    print(f"Beam width: {args.beam_width}")
    print(f"Top-K: {args.top_k}")
    print(f"Feature cache mode: {'required' if args.require_feature_cache else 'optional'}")

    device = torch.device(config.device)

    print("\nLoading vocabulary and models...")
    df_sent = load_sentences(config.train_csv_path, limit=400)
    encodings, reverse_encodings = build_word_dict(df_sent["sign_language"])
    encoder, decoder = load_models(encodings, device)

    total = 0
    top1 = 0
    top3 = 0
    top5 = 0
    missing_label = 0
    missing_features = 0
    errors = 0

    latency_feature = []
    latency_decode = []
    latency_total = []

    rows = []
    class_stats = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0, "top5": 0})

    for idx, video_path in enumerate(videos, start=1):
        video_name = os.path.basename(video_path)
        video_id = os.path.basename(os.path.dirname(video_path))
        expected = label_map.get(str(video_id), None)

        if expected is None:
            missing_label += 1
            continue

        feature_file = os.path.join(args.features_dir, f"{os.path.splitext(video_name)[0]}.pt")
        if args.require_feature_cache and not os.path.exists(feature_file):
            missing_features += 1
            continue

        try:
            t0 = time.perf_counter()
            tf0 = time.perf_counter()
            features = get_video_features(video_path, features_dir=args.features_dir)
            features = preprocess_features(features, device)
            tf1 = time.perf_counter()

            td0 = time.perf_counter()
            hypotheses = decode_sequence_beam(
                encoder, decoder, features, encodings, reverse_encodings, device,
                beam_width=args.beam_width, top_k=max(5, args.top_k)
            )
            td1 = time.perf_counter()
            t1 = time.perf_counter()

            norm_expected = normalize_text(expected)
            nbest = [normalize_text(h[0]) for h in hypotheses]
            pred_top1 = nbest[0] if nbest else ""

            is_top1 = len(nbest) >= 1 and norm_expected == nbest[0]
            is_top3 = norm_expected in nbest[:3]
            is_top5 = norm_expected in nbest[:5]

            total += 1
            top1 += int(is_top1)
            top3 += int(is_top3)
            top5 += int(is_top5)

            class_stats[norm_expected]["total"] += 1
            class_stats[norm_expected]["top1"] += int(is_top1)
            class_stats[norm_expected]["top3"] += int(is_top3)
            class_stats[norm_expected]["top5"] += int(is_top5)

            latency_feature.append(tf1 - tf0)
            latency_decode.append(td1 - td0)
            latency_total.append(t1 - t0)

            rows.append({
                "video_id": video_id,
                "video_name": video_name,
                "video_path": video_path,
                "expected": norm_expected,
                "pred_top1": pred_top1,
                "top1_match": is_top1,
                "top3_match": is_top3,
                "top5_match": is_top5,
                "nbest_json": json.dumps([{"sentence": h[0], "score": float(h[1])} for h in hypotheses], ensure_ascii=False),
                "lat_feature_sec": tf1 - tf0,
                "lat_decode_sec": td1 - td0,
                "lat_total_sec": t1 - t0,
            })

            if idx % 25 == 0:
                print(f"Progress {idx}/{len(videos)} | eval={total} | top1={top1/max(total,1)*100:.2f}%")
        except Exception as e:
            errors += 1
            print(f"[ERROR] {video_name}: {e}")

    results_df = pd.DataFrame(rows)
    results_csv = os.path.join(args.output_dir, "seq2seq_eval_results.csv")
    results_df.to_csv(results_csv, index=False)

    per_class_rows = []
    for label, st in class_stats.items():
        denom = max(st["total"], 1)
        per_class_rows.append({
            "label": label,
            "total": st["total"],
            "top1_acc": 100.0 * st["top1"] / denom,
            "top3_acc": 100.0 * st["top3"] / denom,
            "top5_acc": 100.0 * st["top5"] / denom,
        })
    per_class_df = pd.DataFrame(per_class_rows).sort_values("total", ascending=False)
    per_class_df.to_csv(os.path.join(args.output_dir, "seq2seq_per_class_metrics.csv"), index=False)

    save_confusion_tables(results_df, args.output_dir, top_n=25)

    metrics = {
        "evaluated_videos": total,
        "missing_label": missing_label,
        "missing_features_skipped": missing_features,
        "errors": errors,
        "top1_accuracy": (100.0 * top1 / total) if total else 0.0,
        "top3_accuracy": (100.0 * top3 / total) if total else 0.0,
        "top5_accuracy": (100.0 * top5 / total) if total else 0.0,
        "latency_sec": {
            "feature_mean": (sum(latency_feature) / len(latency_feature)) if latency_feature else 0.0,
            "decode_mean": (sum(latency_decode) / len(latency_decode)) if latency_decode else 0.0,
            "total_mean": (sum(latency_total) / len(latency_total)) if latency_total else 0.0,
            "feature_median": float(pd.Series(latency_feature).median()) if latency_feature else 0.0,
            "decode_median": float(pd.Series(latency_decode).median()) if latency_decode else 0.0,
            "total_median": float(pd.Series(latency_total).median()) if latency_total else 0.0,
        }
    }

    with open(os.path.join(args.output_dir, "seq2seq_eval_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Evaluated videos: {metrics['evaluated_videos']}")
    print(f"Top-1: {metrics['top1_accuracy']:.2f}%")
    print(f"Top-3: {metrics['top3_accuracy']:.2f}%")
    print(f"Top-5: {metrics['top5_accuracy']:.2f}%")
    print(f"Latency mean total (s): {metrics['latency_sec']['total_mean']:.4f}")
    print(f"Saved results: {results_csv}")
    print(f"Saved metrics: {os.path.join(args.output_dir, 'seq2seq_eval_metrics.json')}")


if __name__ == "__main__":
    main()
