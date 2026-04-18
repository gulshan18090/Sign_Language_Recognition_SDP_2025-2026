import torch
import os


class Config:
    """
    Global configuration object.
    Safe to import from any module.
    Contains only simple attributes.
    """

    # --------------------------------------------------
    # Core environment
    # --------------------------------------------------
    env = 'CeDAR'    # Options: 'Dev', 'Prod', 'CeDAR'
    seed = 44
    debug = False

    # --------------------------------------------------
    # Device setup
    # --------------------------------------------------
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # device = 'tpu'   # Uncomment for TPU support

    # --------------------------------------------------
    # Dataset / video loading
    # --------------------------------------------------
    max_frames = 64
    max_words_in_sentence = 10
    BATCH_SIZE = 8

    video_processing_tool = 'TorchVision'   # Options: 'OpenCV', 'VidGear', 'TorchVision'
    camera_source = 'Cam2'                  # Cam1 / Cam2

    # --------------------------------------------------
    # Storage folder paths
    # --------------------------------------------------
    # Default (CeDAR local path)
    drive_folder = 'drive'

    if env == 'Dev':
        drive_folder = 'drive/MyDrive/SLR_test'
    elif env == 'Prod':
        drive_folder = 'drive/MyDrive/SLR/Data'

    video_folder = os.path.join(drive_folder, 'Video')
    train_csv_path = os.path.join(drive_folder, 'sentences_all.csv')

    # --------------------------------------------------
    # Model save paths
    # --------------------------------------------------
    encoder_model_path = os.path.join(drive_folder, 'jamal', 'encoder.model')
    decoder_model_path = os.path.join(drive_folder, 'jamal', 'decoder.model')

    # Ensure save directory exists, but safely (no import loops)
    save_dir = os.path.join(drive_folder, 'jamal')
    os.makedirs(save_dir, exist_ok=True)

    # --------------------------------------------------
    # Video Similarity Settings
    # --------------------------------------------------
    similarity_config = {
        'n_frames': 64,                    # Frames to extract per video
        'max_hands': 2,                    # Maximum hands to track (1 or 2)
        'min_detection_confidence': 0.5,  # MediaPipe detection confidence
        'min_tracking_confidence': 0.5,   # MediaPipe tracking confidence
        'normalize_landmarks': True,       # Normalize for scale/position invariance
        'similarity_method': 'cosine',     # Default: 'cosine', 'euclidean', 'frame_wise_cosine'
        'filter_hand_frames': True,        # Only use frames containing hands
        'feature_dim': 126,                # 21 landmarks × 3 coords × 2 hands
    }


# Export a global instance (common PyTorch practice)
config = Config()
