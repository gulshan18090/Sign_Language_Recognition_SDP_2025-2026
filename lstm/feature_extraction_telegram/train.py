"""
Stage 2: Training on precomputed Telegram features (Videos/ folder)
Uses features from features_telegram/ directory.

This pipeline differs from SLR in how it builds the dataset:
- SLR uses sentence IDs from CSV → numbered video folders
- Telegram uses sentence folder names → video files inside

Usage:
    python train.py                     # Train with defaults
    python train.py --epochs 100        # Custom epochs
"""
import os
import sys

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import argparse
import torch
import pandas as pd
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

from feature_extraction_telegram.config_telegram import (
    FEATURES_DIR, CSV_PATH, MODEL_SAVE_DIR, DEVICE, VIDEO_DIR,
    FEATURE_DIM, TRAIN_HIDDEN_SIZE, BATCH_SIZE, MAX_FRAMES, MAX_WORDS, SEED
)
from data.vocab import build_word_dict
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from utils.seed import seed_everything

# We need to adapt the dataloader and trainer for the telegram format
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import config


def sanitize_filename(name):
    """Convert a sentence folder name to a safe filename (must match extract_features.py)."""
    safe = name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe = safe.replace('?', '_').replace('!', '_').replace(';', '_')
    safe = safe.replace(',', '_').replace('(', '_').replace(')', '_')
    safe = safe.replace("'", '_').replace('"', '_')
    while '__' in safe:
        safe = safe.replace('__', '_')
    return safe.strip('_')


class TelegramFeatureDataset(Dataset):
    """Dataset for telegram features organized by sentence folders."""

    def __init__(self, df, features_dir):
        self.df = df.reset_index(drop=True)
        self.features_dir = features_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_file = row["feature_file"]
        encoding = torch.tensor(row["encoding"])
        enc_len = encoding.shape[0]

        feature_path = os.path.join(self.features_dir, feature_file)
        if not os.path.exists(feature_path):
            return None, None, None

        features = torch.load(feature_path, map_location='cpu', weights_only=True)
        return features, encoding.reshape(enc_len, 1), row.get("sentence", feature_file)


def pad_collate_features(batch):
    """Pad features and labels to uniform length."""
    # Filter out None entries
    batch = [(f, l, n) for f, l, n in batch if f is not None]
    if not batch:
        return None, None, None

    features, labels, fnames = zip(*batch)

    target_length = MAX_FRAMES
    padded_f = []
    for f in features:
        seq_len = f.shape[0]
        if seq_len < target_length:
            pad = torch.zeros((target_length - seq_len, f.shape[1]), dtype=f.dtype)
            f = torch.cat([f, pad], dim=0)
        elif seq_len > target_length:
            indices = torch.linspace(0, seq_len - 1, target_length).long()
            f = f[indices]
        padded_f.append(f)
    features = torch.stack(padded_f)

    max_lab_len = max(lbl.shape[0] for lbl in labels)
    padded_l = []
    for lbl in labels:
        if lbl.shape[0] < max_lab_len:
            pad = torch.zeros((max_lab_len - lbl.shape[0], 1), dtype=lbl.dtype)
            lbl = torch.cat([lbl, pad], dim=0)
        padded_l.append(lbl)
    labels = torch.stack(padded_l)

    return features, labels, fnames


def build_telegram_video_table(video_dir, features_dir, encodings):
    """
    Build video table from Videos/ folder structure.
    Maps sentence folders → feature files → encodings.
    """
    from data.vocab import encode_sentence

    rows = []

    for sentence_folder in sorted(os.listdir(video_dir)):
        folder_path = os.path.join(video_dir, sentence_folder)
        if not os.path.isdir(folder_path):
            continue

        video_files = [f for f in os.listdir(folder_path)
                       if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        if not video_files:
            continue

        sentence_text = sentence_folder.strip().lower()
        encoded = encode_sentence(sentence_text, encodings)
        safe_sentence = sanitize_filename(sentence_folder)

        for video_file in video_files:
            video_name = os.path.splitext(video_file)[0]
            feature_name = f"{safe_sentence}__{sanitize_filename(video_name)}.pt"
            feature_path = os.path.join(features_dir, feature_name)

            if os.path.exists(feature_path):
                rows.append({
                    'idd': sentence_folder,
                    'sentence': sentence_folder,
                    'sign_language': sentence_text,
                    'feature_file': feature_name,
                    'encoding': encoded,
                    'video_file': os.path.join(folder_path, video_file)
                })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Telegram Training Pipeline (Videos/)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--history-path", type=str,
                        default=os.path.join(PARENT_DIR, "research_outputs", "telegram_training_history.csv"))
    args = parser.parse_args()

    seed_everything(SEED)

    # Build dataset CSV if not exists
    if not os.path.exists(CSV_PATH):
        print("[INFO] Building telegram dataset CSV...")
        from feature_extraction_telegram.build_dataset import build_telegram_csv
        original_csv = os.path.join(PARENT_DIR, "drive", "sentences_all.csv")
        build_telegram_csv(VIDEO_DIR, original_csv=original_csv, output_path=CSV_PATH)

    # Load CSV and build vocab
    df_sent = pd.read_csv(
        CSV_PATH, sep=';', encoding='utf-8',
        header=None, names=['idd', 'sentence', 'sign_language']
    )
    encodings, word_idx = build_word_dict(df_sent["sign_language"])

    # Build video table with feature file mapping
    df = build_telegram_video_table(VIDEO_DIR, FEATURES_DIR, encodings)

    if len(df) == 0:
        print(f"❌ No features found in {FEATURES_DIR}")
        print("Run extract_features.py first!")
        return

    print(f"[INFO] Training on {len(df)} samples with features from {FEATURES_DIR}")
    print(f"[INFO] Unique sentences: {df['sentence'].nunique()}")

    # Apply data fraction
    if args.data_fraction < 1.0:
        before = len(df)
        sampled_parts = []
        for _, group in df.groupby("idd", sort=False):
            keep_n = max(1, int(len(group) * args.data_fraction))
            sampled_parts.append(group.sample(n=keep_n, random_state=SEED))
        df = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        print(f"[INFO] Data fraction {args.data_fraction:.2f}: {before} -> {len(df)} samples")

    # Ensure model save directory exists
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

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
    print(f"Telegram Training Pipeline (Videos/)")
    print(f"  Features: {FEATURES_DIR}")
    print(f"  Samples: {len(df)}")
    print(f"  Unique sentences: {df['sentence'].nunique()}")
    print(f"  Vocab size: {len(encodings)}")
    print(f"  Feature dim: {FEATURE_DIM}")
    print(f"  Hidden size: {TRAIN_HIDDEN_SIZE}")
    print(f"  Epochs: {args.epochs}")
    print(f"{'='*60}\n")

    # --- Custom Training Loop (adapted from models/trainer.py) ---
    import time
    import gc
    import csv
    import numpy as np
    import torch.optim as optim
    from models.train_step import train_step

    enc_opt = optim.Adam(encoder.parameters(), lr=0.01)
    dec_opt = optim.Adam(decoder.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        enc_opt, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    )
    criterion = torch.nn.CrossEntropyLoss()

    # Split into train/val
    unique_sentences = df['sentence'].unique()
    if len(unique_sentences) > 1:
        # Group by sentence for stratified split
        sentence_groups = df.groupby('sentence')
        train_dfs = []
        val_dfs = []
        for sent, group in sentence_groups:
            if len(group) >= 2:
                tr, va = train_test_split(group, test_size=0.1, random_state=SEED)
                train_dfs.append(tr)
                val_dfs.append(va)
            else:
                train_dfs.append(group)
        train_df = pd.concat(train_dfs).reset_index(drop=True)
        val_df = pd.concat(val_dfs).reset_index(drop=True) if val_dfs else train_df.sample(frac=0.1, random_state=SEED)
    else:
        train_df, val_df = train_test_split(df, test_size=0.1, random_state=SEED)

    train_dataset = TelegramFeatureDataset(train_df, FEATURES_DIR)
    val_dataset = TelegramFeatureDataset(val_df, FEATURES_DIR)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=pad_collate_features)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, collate_fn=pad_collate_features)

    print(f"[INFO] Training batches: {len(train_loader)}")
    print(f"[INFO] Validation batches: {len(val_loader)}")

    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    history_rows = []

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        epoch_start = time.time()
        epoch_train_losses = []

        encoder.train()
        decoder.train()

        for i, (x, y, fname) in enumerate(train_loader, start=1):
            if x is None:
                continue

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            try:
                loss = train_step(
                    x, y,
                    encoder, decoder,
                    enc_opt, dec_opt,
                    criterion, encodings,
                    teacher_forcing_ratio=0.5,
                    epoch=epoch,
                    total_epochs=args.epochs
                )
                epoch_train_losses.append(loss)
            except Exception as e:
                print(f"⚠️ Error: {e}")
                continue

            if i % 10 == 0 or i == len(train_loader):
                print(f"[{i}/{len(train_loader)}] Loss: {loss:.4f}")

            # Periodic save
            if i % args.print_every == 0:
                torch.save(encoder.state_dict(), os.path.join(MODEL_SAVE_DIR, "encoder.model"))
                torch.save(decoder.state_dict(), os.path.join(MODEL_SAVE_DIR, "decoder.model"))

        avg_train_loss = np.mean(epoch_train_losses) if epoch_train_losses else float('inf')

        # Validation
        encoder.eval()
        decoder.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, fname in val_loader:
                if x is None:
                    continue
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                try:
                    B = x.size(0)
                    hidden = encoder.initHidden(B)
                    encoder_output, encoder_hidden = encoder(x, hidden)

                    if encoder.D == 2:
                        h = torch.cat((encoder_hidden[0][0:1], encoder_hidden[0][1:2]), dim=2)
                        c = torch.cat((encoder_hidden[1][0:1], encoder_hidden[1][1:2]), dim=2)
                        decoder_hidden = (h, c)
                    else:
                        decoder_hidden = encoder_hidden

                    max_len = y.shape[1]
                    decoder_input = y[:, :max_len-2, :].squeeze(-1)
                    decoder_target = y[:, 1:max_len-1, :].squeeze(-1)

                    loss = 0
                    for t in range(decoder_target.size(1)):
                        output, decoder_hidden, _ = decoder(
                            decoder_input[:, t].unsqueeze(1), decoder_hidden, encoder_output
                        )
                        tgt = decoder_target[:, t]
                        loss += criterion(output, tgt)

                    val_losses.append(loss.item() / decoder_target.size(1))
                except Exception:
                    continue

        val_loss = np.mean(val_losses) if val_losses else float('inf')
        scheduler.step(val_loss)
        for pg in dec_opt.param_groups:
            pg['lr'] = enc_opt.param_groups[0]['lr']
        current_lr = enc_opt.param_groups[0]['lr']

        print(f"📊 Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}")

        history_rows.append({
            "epoch": epoch + 1,
            "train_loss": float(avg_train_loss),
            "val_loss": float(val_loss),
            "learning_rate": float(current_lr),
        })

        # Early stopping
        if val_loss < best_val_loss - args.min_delta:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(encoder.state_dict(), os.path.join(MODEL_SAVE_DIR, "encoder_best.model"))
            torch.save(decoder.state_dict(), os.path.join(MODEL_SAVE_DIR, "decoder_best.model"))
            print(f"✅ New best model! Val Loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ No improvement. Patience: {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"\n🛑 Early stopping! Best epoch: {best_epoch}, Val Loss: {best_val_loss:.4f}")
                encoder.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, "encoder_best.model")))
                decoder.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, "decoder_best.model")))
                torch.save(encoder.state_dict(), os.path.join(MODEL_SAVE_DIR, "encoder.model"))
                torch.save(decoder.state_dict(), os.path.join(MODEL_SAVE_DIR, "decoder.model"))
                break

        print(f"⏱️ Epoch time: {time.time() - epoch_start:.1f}s")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n✅ Training completed! Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    # Save history
    if args.history_path and history_rows:
        os.makedirs(os.path.dirname(args.history_path) or ".", exist_ok=True)
        with open(args.history_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)
        print(f"📝 Saved training history: {args.history_path}")


if __name__ == "__main__":
    main()
