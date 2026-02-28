"""
Run data-efficiency experiments for seq2seq SLR.

For each fraction (e.g., 0.25, 0.50, 0.75, 1.00), this script:
1) Trains using main.py with that data fraction
2) Copies trained model checkpoints to fraction-specific filenames
3) Runs unified evaluation and stores metrics/results
4) Writes a summary CSV for plotting
"""
import os
import csv
import json
import shutil
import argparse
import subprocess


def run_cmd(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def copy_if_exists(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Data-efficiency study runner")
    parser.add_argument("--fractions", default="0.25,0.50,0.75,1.00",
                        help="Comma-separated fractions in (0,1].")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs for each fraction.")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience.")
    parser.add_argument("--min-delta", type=float, default=0.001, help="Early stopping min delta.")
    parser.add_argument("--limit", type=int, default=400, help="Sentence limit for training/vocab.")
    parser.add_argument("--print-every", type=int, default=50, help="Checkpoint save frequency.")
    parser.add_argument("--eval-max-videos", type=int, default=0,
                        help="Evaluation subset size (0 means all videos).")
    parser.add_argument("--beam-width", type=int, default=5, help="Beam width for evaluation.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k for eval metrics.")
    parser.add_argument("--require-feature-cache", action="store_true",
                        help="Skip evaluation samples without feature cache.")
    parser.add_argument("--output-dir", default="research_outputs/data_efficiency", help="Output directory.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fractions = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
    for f in fractions:
        if f <= 0 or f > 1:
            raise ValueError(f"Invalid fraction: {f}. Must be in (0,1].")

    summary_rows = []

    for fraction in fractions:
        tag = f"f{int(round(fraction * 100)):03d}"
        exp_dir = os.path.join(args.output_dir, tag)
        os.makedirs(exp_dir, exist_ok=True)

        # 1) Train
        train_cmd = [
            "python3", "main.py",
            "--data-fraction", str(fraction),
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--min-delta", str(args.min_delta),
            "--limit", str(args.limit),
            "--print-every", str(args.print_every),
            "--history-path", os.path.join(exp_dir, "training_history.csv"),
        ]
        run_cmd(train_cmd)

        # 2) Snapshot trained models
        model_dir = "drive/jamal"
        enc_src = os.path.join(model_dir, "encoder.model")
        dec_src = os.path.join(model_dir, "decoder.model")
        enc_best_src = os.path.join(model_dir, "encoder_best.model")
        dec_best_src = os.path.join(model_dir, "decoder_best.model")

        copy_if_exists(enc_src, os.path.join(exp_dir, "encoder.model"))
        copy_if_exists(dec_src, os.path.join(exp_dir, "decoder.model"))
        copy_if_exists(enc_best_src, os.path.join(exp_dir, "encoder_best.model"))
        copy_if_exists(dec_best_src, os.path.join(exp_dir, "decoder_best.model"))

        # 3) Evaluate
        eval_cmd = [
            "python3", "research_eval_seq2seq.py",
            "--output-dir", exp_dir,
            "--beam-width", str(args.beam_width),
            "--top-k", str(args.top_k),
            "--max-videos", str(args.eval_max_videos),
        ]
        if args.require_feature_cache:
            eval_cmd.append("--require-feature-cache")
        run_cmd(eval_cmd)

        # 4) Read metrics
        metrics_path = os.path.join(exp_dir, "seq2seq_eval_metrics.json")
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        summary_rows.append({
            "fraction": fraction,
            "evaluated_videos": metrics.get("evaluated_videos", 0),
            "top1_accuracy": metrics.get("top1_accuracy", 0.0),
            "top3_accuracy": metrics.get("top3_accuracy", 0.0),
            "top5_accuracy": metrics.get("top5_accuracy", 0.0),
            "latency_total_mean_sec": metrics.get("latency_sec", {}).get("total_mean", 0.0),
            "latency_decode_mean_sec": metrics.get("latency_sec", {}).get("decode_mean", 0.0),
            "latency_feature_mean_sec": metrics.get("latency_sec", {}).get("feature_mean", 0.0),
            "errors": metrics.get("errors", 0),
        })

    summary_csv = os.path.join(args.output_dir, "data_efficiency_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 80)
    print("DATA EFFICIENCY STUDY COMPLETE")
    print("=" * 80)
    print(f"Saved summary: {summary_csv}")
    for row in summary_rows:
        print(
            f"fraction={row['fraction']:.2f} | "
            f"top1={row['top1_accuracy']:.2f}% | "
            f"top3={row['top3_accuracy']:.2f}% | "
            f"top5={row['top5_accuracy']:.2f}%"
        )


if __name__ == "__main__":
    main()
