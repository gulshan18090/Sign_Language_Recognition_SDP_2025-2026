"""
Generate research paper figures from evaluation/training artifacts.
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def plot_learning_curve(history_csv, out_png):
    df = pd.read_csv(history_csv)
    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], marker="o", label="Train Loss")
    plt.plot(df["epoch"], df["val_loss"], marker="s", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_confusion_heatmap(confusion_csv, out_png, max_classes=20):
    df = pd.read_csv(confusion_csv, index_col=0)
    if df.shape[0] > max_classes:
        top_rows = df.sum(axis=1).sort_values(ascending=False).head(max_classes).index
        df = df.loc[top_rows]
    if df.shape[1] > max_classes:
        top_cols = df.sum(axis=0).sort_values(ascending=False).head(max_classes).index
        df = df[top_cols]

    plt.figure(figsize=(10, 8))
    plt.imshow(df.values, aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(range(len(df.columns)), df.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(df.index)), df.index, fontsize=7)
    plt.title("Top-Class Confusion Matrix (Top-1)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot learning curve and confusion matrix.")
    parser.add_argument("--history-csv", required=True, help="Path to training history CSV")
    parser.add_argument("--confusion-csv", required=True, help="Path to confusion matrix CSV")
    parser.add_argument("--output-dir", default="research_outputs/figures", help="Output figure directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    learning_png = os.path.join(args.output_dir, "learning_curve.png")
    confusion_png = os.path.join(args.output_dir, "confusion_matrix_top_classes.png")

    plot_learning_curve(args.history_csv, learning_png)
    plot_confusion_heatmap(args.confusion_csv, confusion_png)

    print(f"Saved: {learning_png}")
    print(f"Saved: {confusion_png}")


if __name__ == "__main__":
    main()
