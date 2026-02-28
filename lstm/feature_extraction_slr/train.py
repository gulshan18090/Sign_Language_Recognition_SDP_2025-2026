"""
Stage 2: Training on precomputed SLR features (drive/Video/Cam2)
Uses features from features_slr/ directory.

Usage:
    python train.py                     # Train with defaults
    python train.py --epochs 100        # Custom epochs
    python train.py --limit 200         # Limit CSV rows
"""
import os
import sys

# Add parent directory to path for shared modules
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from feature_extraction_slr.config_slr import (
    FEATURES_DIR, CSV_PATH, MODEL_SAVE_DIR, DEVICE,
    FEATURE_DIM, TRAIN_HIDDEN_SIZE, BATCH_SIZE, SEED, CAMERA_SOURCE
)
from config import config
from data.vocab import build_word_dict
from data.dataset_builder import load_sentences, build_video_table
from data.dataloader import get_feature_dataloader
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from models.trainer import trainIters
from utils.seed import seed_everything


def main():
    parser = argparse.ArgumentParser(description="SLR Training Pipeline (Cam2)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--history-path", type=str,
                        default=os.path.join(PARENT_DIR, "research_outputs", "slr_training_history.csv"))
    args = parser.parse_args()

    seed_everything(SEED)

    # Override config camera source for this pipeline
    config.camera_source = CAMERA_SOURCE

    # Load CSV and build vocab
    df_sent = load_sentences(CSV_PATH, limit=args.limit)
    encodings, word_idx = build_word_dict(df_sent["sign_language"])

    # Build video table
    df = build_video_table(df_sent, encodings)

    # Filter to only samples with available features
    if os.path.exists(FEATURES_DIR):
        available_features = set(f[:-3] for f in os.listdir(FEATURES_DIR) if f.endswith(".pt"))
        initial_count = len(df)
        df = df[df["video_file"].apply(
            lambda x: os.path.splitext(os.path.basename(x).replace(' ', '_'))[0] in available_features
        )].reset_index(drop=True)
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"[INFO] Filtered out {filtered_count} samples with missing features.")
        print(f"[INFO] Training on {len(df)} samples with features from {FEATURES_DIR}")
    else:
        print(f"❌ Features directory not found: {FEATURES_DIR}")
        print("Run extract_features.py first!")
        return

    # Apply data fraction
    if args.data_fraction < 1.0:
        import pandas as pd
        before = len(df)
        sampled_parts = []
        for _, group in df.groupby("idd", sort=False):
            keep_n = max(1, int(len(group) * args.data_fraction))
            sampled_parts.append(group.sample(n=keep_n, random_state=SEED))
        df = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        print(f"[INFO] Data fraction {args.data_fraction:.2f}: {before} -> {len(df)} samples")

    # Initialize models
    encoder = EncoderRNN(
        FEATURE_DIM, TRAIN_HIDDEN_SIZE,
        device=DEVICE, biDirectional=True, use_cnn=False
    ).to(DEVICE)

    encoder_output_size = TRAIN_HIDDEN_SIZE * 2  # Bidirectional
    decoder = AttnDecoderRNN(
        encoder_output_size, len(encodings),
        device=DEVICE,
        encoder_hidden_size=encoder_output_size,
        attn_dropout_p=0.1,
        attn_temperature=2.0
    ).to(DEVICE)

    print(f"\n{'='*60}")
    print(f"SLR Training Pipeline (Cam2)")
    print(f"  Features: {FEATURES_DIR}")
    print(f"  Samples: {len(df)}")
    print(f"  Vocab size: {len(encodings)}")
    print(f"  Feature dim: {FEATURE_DIM}")
    print(f"  Hidden size: {TRAIN_HIDDEN_SIZE}")
    print(f"  Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # Train
    trainIters(
        df, encoder, decoder, encodings,
        print_every=args.print_every,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        history_path=args.history_path
    )


if __name__ == "__main__":
    main()
