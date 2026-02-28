"""
Inference for SLR pipeline (Cam2 features)

Usage:
    python inference.py                         # Run inference on all features
    python inference.py --video path/to/video   # Run on a specific video
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

from feature_extraction_slr.config_slr import (
    FEATURES_DIR, CSV_PATH, MODEL_SAVE_DIR, DEVICE,
    FEATURE_DIM, TRAIN_HIDDEN_SIZE, MAX_FRAMES, SEED
)
from data.vocab import build_word_dict
from data.dataset_builder import load_sentences
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN


def load_models(encodings):
    """Load trained encoder and decoder."""
    encoder = EncoderRNN(
        FEATURE_DIM, TRAIN_HIDDEN_SIZE,
        device=DEVICE, biDirectional=True, use_cnn=False
    ).to(DEVICE)

    encoder_output_size = TRAIN_HIDDEN_SIZE * 2
    decoder = AttnDecoderRNN(
        encoder_output_size, len(encodings),
        device=DEVICE,
        encoder_hidden_size=encoder_output_size,
        attn_dropout_p=0.1,
        attn_temperature=2.0
    ).to(DEVICE)

    enc_path = os.path.join(MODEL_SAVE_DIR, "encoder_best.model")
    dec_path = os.path.join(MODEL_SAVE_DIR, "decoder_best.model")

    if not os.path.exists(enc_path):
        enc_path = os.path.join(MODEL_SAVE_DIR, "encoder.model")
        dec_path = os.path.join(MODEL_SAVE_DIR, "decoder.model")

    encoder.load_state_dict(torch.load(enc_path, map_location=DEVICE))
    decoder.load_state_dict(torch.load(dec_path, map_location=DEVICE))

    encoder.eval()
    decoder.eval()
    return encoder, decoder


def predict(features, encoder, decoder, word_idx, max_words=10):
    """Run inference on pre-extracted features."""
    with torch.no_grad():
        features = features.to(DEVICE)
        if features.dim() == 2:
            features = features.unsqueeze(0)

        # Pad/truncate to MAX_FRAMES
        seq_len = features.shape[1]
        if seq_len < MAX_FRAMES:
            pad = torch.zeros((1, MAX_FRAMES - seq_len, features.shape[2]),
                              device=DEVICE)
            features = torch.cat([features, pad], dim=1)
        elif seq_len > MAX_FRAMES:
            indices = torch.linspace(0, seq_len - 1, MAX_FRAMES).long()
            features = features[:, indices, :]

        hidden = encoder.initHidden(1)
        encoder_output, encoder_hidden = encoder(features, hidden)

        # Prepare decoder hidden
        h = torch.cat((encoder_hidden[0][0:1], encoder_hidden[0][1:2]), dim=2)
        c = torch.cat((encoder_hidden[1][0:1], encoder_hidden[1][1:2]), dim=2)
        decoder_hidden = (h, c)

        # Start with SOS token
        sos_idx = next((i for w, i in word_idx.items() if w == 'SOS'), 0)
        eos_idx = next((i for w, i in word_idx.items() if w == 'EOS'), 1)
        decoder_input = torch.tensor([[sos_idx]], device=DEVICE)

        words = []
        for _ in range(max_words):
            output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_output)
            predicted_idx = output.argmax(dim=1).item()

            if predicted_idx == eos_idx:
                break

            # Lookup word
            reverse_idx = {v: k for k, v in word_idx.items()}
            word = reverse_idx.get(predicted_idx, f"<{predicted_idx}>")
            words.append(word)

            decoder_input = torch.tensor([[predicted_idx]], device=DEVICE)

        return " ".join(words)


def main():
    parser = argparse.ArgumentParser(description="SLR Inference (Cam2)")
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    # Build vocab
    df_sent = load_sentences(CSV_PATH, limit=args.limit)
    encodings, word_idx = build_word_dict(df_sent["sign_language"])

    # Load models
    encoder, decoder = load_models(encodings)
    print(f"Models loaded from {MODEL_SAVE_DIR}")

    # Run inference on all features
    if not os.path.exists(FEATURES_DIR):
        print(f"❌ Features dir not found: {FEATURES_DIR}")
        return

    feature_files = sorted([f for f in os.listdir(FEATURES_DIR) if f.endswith(".pt")])
    print(f"\nRunning inference on {len(feature_files)} features...\n")

    results = []
    for fname in feature_files:
        fpath = os.path.join(FEATURES_DIR, fname)
        features = torch.load(fpath, map_location=DEVICE, weights_only=True)
        prediction = predict(features, encoder, decoder, encodings)
        video_id = fname.replace(".pt", "")
        results.append({"video_id": video_id, "prediction": prediction})
        print(f"  {video_id}: {prediction}")

    # Save results
    results_df = pd.DataFrame(results)
    out_path = os.path.join(PARENT_DIR, "slr_inference_results.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\n✅ Results saved to {out_path}")


if __name__ == "__main__":
    main()
