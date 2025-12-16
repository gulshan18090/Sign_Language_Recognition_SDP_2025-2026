import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Only show ERROR logs, hide INFO/DEBUG
import torch
import os
import warnings
from config import Config as config
from data.vocab import build_word_dict
from data.dataset_builder import load_sentences, build_video_table
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from models.trainer import trainIters
from utils.seed import seed_everything
import absl.logging
warnings.filterwarnings("ignore")
# Configure absl logging to suppress logs
# Suppress Abseil logs
absl.logging.get_absl_handler().python_handler.stream = open(os.devnull, 'w')
absl.logging.set_verbosity(absl.logging.FATAL)
absl.logging.set_stderrthreshold(absl.logging.FATAL)


# Suppress unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  
os.environ['TF_ABSL_LOG_LEVEL'] = '3'  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*pts_unit.*")

import sys
import argparse
from data.dataloader import get_feature_dataloader

def main():
    parser = argparse.ArgumentParser(description="SLR Training Pipeline")
    parser.add_argument("--only", action="store_true", help="Train using only available features in features/ folder")
    args = parser.parse_args()

    seed_everything(config.seed)

    # Load CSV and build vocab (load all for full vocabulary)
    df_sent = load_sentences(config.train_csv_path, limit=400)
    encodings, word_idx = build_word_dict(df_sent["sign_language"])

    # Build video table
    df = build_video_table(df_sent, encodings)

    # Filter df to only those with available features (since we're using feature dataloader)
    features_dir = "features"
    if os.path.exists(features_dir):
        available_features = set(f[:-3] for f in os.listdir(features_dir) if f.endswith(".pt"))
        initial_count = len(df)
        df = df[df["video_file"].apply(lambda x: os.path.splitext(os.path.basename(x))[0] in available_features)].reset_index(drop=True)
        filtered_count = initial_count - len(df)
        if filtered_count > 0:
            print(f"[INFO] Filtered out {filtered_count} samples with missing features. Training on {len(df)} samples.")
    else:
        print(f"⚠️ Warning: Features directory '{features_dir}' not found. Training may fail.")

    # Initialize models
    # Features are pre-extracted with dimension 512 (from extract_features.py: hidden=256, bidirectional)
    input_size = 512  # Pre-extracted feature dimension (256 * 2 from bidirectional)
    hidden_size = 256  # Hidden size for training encoder/decoder (larger = more capacity)

    # use_cnn=False: We're using pre-extracted features, no need for SqueezeNet in training
    # biDirectional=True for better accuracy (5-10% improvement)
    encoder = EncoderRNN(input_size, hidden_size, device=config.device, biDirectional=True, use_cnn=False).to(config.device)
    # Encoder output is 256*2=512 for bidirectional
    encoder_output_size = hidden_size * 2  # 512
    decoder = AttnDecoderRNN(encoder_output_size, len(encodings), device=config.device, encoder_hidden_size=encoder_output_size).to(config.device)

    # Train the model with early stopping
    # epochs=20: Maximum epochs (will stop early if no improvement)
    # patience=5: Stop if no improvement for 5 epochs
    # min_delta=0.001: Minimum change to qualify as improvement
    trainIters(df, encoder, decoder, encodings, 
               print_every=50, 
               epochs=20, 
               patience=7, 
               min_delta=0.001)

if __name__ == "__main__":
    main()

