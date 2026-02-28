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
    BATCH_SIZE = 64

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


# Export a global instance (common PyTorch practice)
config = Config()
