"""
Config for Telegram Feature Extraction Pipeline (Videos/ folder)
Videos collected via Telegram bot, organized by sentence name folders.
"""
import os
import sys

# Add parent directory to path for shared modules
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import torch

# ----- Paths -----
VIDEO_DIR = os.path.join(PARENT_DIR, "Videos")
FEATURES_DIR = os.path.join(PARENT_DIR, "features_telegram")
CSV_PATH = os.path.join(PARENT_DIR, "feature_extraction_telegram", "sentences_telegram.csv")
MODEL_SAVE_DIR = os.path.join(PARENT_DIR, "feature_extraction_telegram", "models_telegram")

# ----- Device -----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Model hyperparams -----
CNN_FEATURE_SIZE = 86528       # SqueezeNet flattened output (512*13*13)
ENCODER_HIDDEN_SIZE = 256      # LSTM hidden size in extraction encoder
BIDIRECTIONAL = True           # Bidirectional LSTM
FEATURE_DIM = ENCODER_HIDDEN_SIZE * (2 if BIDIRECTIONAL else 1)  # 512

# ----- Training hyperparams -----
TRAIN_HIDDEN_SIZE = 512
BATCH_SIZE = 64
MAX_FRAMES = 64
MAX_WORDS = 10
SEED = 44

# ----- Preprocessing -----
CROP_SIZE = 600
RESIZE = 224

print(f"[Telegram Config] Video dir: {VIDEO_DIR}")
print(f"[Telegram Config] Features dir: {FEATURES_DIR}")
print(f"[Telegram Config] Device: {DEVICE}")
