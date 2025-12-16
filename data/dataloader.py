import os
import torch
import cv2
import sklearn.utils
import torchvision
import logging
import absl.logging
import warnings
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from video.mp_hands import keep_frames_with_hands
from video.transforms import apply_video_transforms
from config import config
from sklearn.model_selection import train_test_split
mp.set_start_method('spawn', force=True)

# -------------------- Fast Feature Dataset for Stage 2 --------------------
class FeatureDataset(Dataset):
    def __init__(self, df, features_dir):
        self.df = df.reset_index(drop=True)
        self.features_dir = features_dir
        # Note: Features are loaded on-demand to avoid memory issues with large datasets
        # For small datasets, consider pre-loading all features in __init__ for faster training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_file"]
        encoding = torch.tensor(row["encoding"])
        enc_len = encoding.shape[0]

        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Load pre-extracted features from disk
        feature_file_path = os.path.join(self.features_dir, f"{video_name}.pt")

        if not os.path.exists(feature_file_path):
            # Log warning and skip missing feature files
            logging.warning(f"Feature file not found: {feature_file_path}. Skipping sample.")
            return None, None, None

        # Load features and return on CPU - DataLoader will handle device transfer
        features = torch.load(feature_file_path, map_location='cpu', weights_only=True)
        return features, encoding.reshape(enc_len, 1), video_path


# ----------------------------------------------------------
# Feature Dataloader creator for Stage 2
# ----------------------------------------------------------
def get_feature_dataloader(df, phase, batch_size, features_dir="features"):
    # Filter out classes with fewer than 2 samples
    class_counts = df['idd'].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df['idd'].isin(valid_classes)]

    # Perform train-test split
    if len(valid_classes) > 1:
        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=config.seed, stratify=df['idd']
        )
    else:
        # Fallback to random splitting if stratified splitting is not possible
        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=config.seed
        )

    if phase == "train":
        dataset = FeatureDataset(train_df, features_dir)
    else:
        dataset = FeatureDataset(val_df, features_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate_features

    )
    return loader

# Set logging to suppress unnecessary logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
absl.logging.set_verbosity(absl.logging.ERROR)  # Set absl logging to ERROR

# Suppress MediaPipe logs
logging.getLogger("mediapipe").setLevel(logging.ERROR)  # Set MediaPipe logs to ERROR only

# Suppress warnings in general
warnings.filterwarnings("ignore")

# Ensure we are using the GPU (if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# -------------------- Data Processing Functions --------------------
def save_video_features(video_path, features_dir):
    """
    Process a video, extract frames and save the features as tensors.
    """
    # Ensure directory for saving features exists
    os.makedirs(features_dir, exist_ok=True)

    # Extract frames using the defined function
    reader = cv2.VideoCapture(video_path)
    frames = keep_frames_with_hands(reader, crop_size=600)

    if frames is None or frames.shape[0] == 0:
        print(f"⚠️ No valid frames found in {video_path}")
        return

    # Process the frames (e.g., apply transformations)
    frames = frames.to(device)

    # Feature extraction (pass frames through EncoderRNN or similar)
    feature_tensor = feature_extractor(frames)  # Assuming `feature_extractor` is defined elsewhere
    feature_tensor = feature_tensor.cpu()  # Ensure tensor is on CPU before saving

    # Save the tensor to disk
    video_name = os.path.basename(video_path).split('.')[0]
    feature_path = os.path.join(features_dir, f"{video_name}_features.pt")
    torch.save(feature_tensor, feature_path)

    print(f"Saved features for {video_name} to {feature_path}")

def pad_collate_features(batch):
    features, labels, fnames = zip(*batch)

    # --- Pad/truncate features to config.max_frames ---
    target_length = config.max_frames
    padded_f = []
    for f in features:
        seq_len = f.shape[0]
        if seq_len < target_length:
            # Pad to target_length
            pad_len = target_length - seq_len
            pad = torch.zeros((pad_len, f.shape[1]), dtype=f.dtype, device=f.device)
            f = torch.cat([f, pad], dim=0)
        elif seq_len > target_length:
            # Truncate to target_length (take middle frames or evenly sample)
            # Option 1: Take first target_length frames
            f = f[:target_length]
            # Option 2: Take evenly spaced frames (uncomment if preferred)
            # indices = torch.linspace(0, seq_len - 1, target_length).long()
            # f = f[indices]
        padded_f.append(f)
    features = torch.stack(padded_f)

    # --- Pad labels (word sequences) ---
    max_lab_len = max(lbl.shape[0] for lbl in labels)
    padded_l = []
    for lbl in labels:
        if lbl.shape[0] < max_lab_len:
            pad_len = max_lab_len - lbl.shape[0]
            pad = torch.zeros((pad_len, 1), dtype=lbl.dtype, device=lbl.device)
            lbl = torch.cat([lbl, pad], dim=0)
        padded_l.append(lbl)
    labels = torch.stack(padded_l)

    return features, labels, fnames


def pad_collate(batch):
    """
    Padding videos to the maximum sequence length in the batch
    """
    videos, labels, fnames = zip(*batch)

    lengths = [v.shape[0] for v in videos]
    max_len = max(lengths)

    padded_videos = []
    for v in videos:
        pad_len = max_len - v.shape[0]
        if pad_len > 0:
            pad = torch.zeros([pad_len, *v.shape[1:]], dtype=v.dtype, device=device)  # Move to GPU
            v = torch.cat([v, pad], dim=0)
        padded_videos.append(v)

    padded_videos = torch.stack(padded_videos).to(device)  # Move to GPU
    labels = torch.stack(labels).to(device)  # Move to GPU

    return padded_videos, labels, fnames

# -------------------- Dataset Class --------------------
class SLDataset(Dataset):
    def __init__(self, df, features_dir=None, video_transform=True):
        self.df = df.reset_index(drop=True)
        self.features_dir = features_dir
        self.video_transform = video_transform
        self.device = device  # Ensure GPU usage

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = row["video_file"]
        encoding = torch.tensor(row["encoding"]).to(device)  # Move encoding to GPU early
        enc_len = encoding.shape[0]

        if self.features_dir is not None:
            # Save the features if not already saved
            video_name = os.path.basename(video_path).split('.')[0]
            feature_file_path = os.path.join(self.features_dir, f"{video_name}_features.pt")

            if not os.path.exists(feature_file_path):
                save_video_features(video_path, self.features_dir)

            # Load the saved feature tensor
            features = torch.load(feature_file_path).to(device)
            return features, encoding.reshape(enc_len, 1), video_path
        else:
            # Direct video processing (no feature caching)
            # ---------- Load video ----------
            if config.video_processing_tool == "OpenCV":
                reader = cv2.VideoCapture(video_path)
            elif config.video_processing_tool == "VidGear":
                reader = CamGear(source=video_path).start()
            elif config.video_processing_tool == "TorchVision":
                reader, _, _ = torchvision.io.read_video(video_path, output_format="THWC", pts_unit='sec')

            # ---------- Extract frames ----------
            frames = keep_frames_with_hands(reader, crop_size=600)

            if frames is None or frames.shape[0] == 0:
                empty = torch.zeros((config.max_frames, 3, 224, 224), device=device)
                return empty, encoding.reshape(enc_len, 1), video_path

            # Ensure shape (N,C,H,W)
            if frames.ndim == 4 and frames.shape[1] not in [1, 3]:
                frames = frames.permute(0, 2, 1, 3)

            frames = frames.to(device)  # Move frames to GPU early

            # Apply transforms (on GPU)
            if self.video_transform:
                frames = apply_video_transforms()(frames)

            # Adjust to max_frames
            n = frames.shape[0]
            if n > config.max_frames and n < 2 * config.max_frames:
                left = (n - config.max_frames) // 2
                frames = frames[left:left + config.max_frames]
            elif n > config.max_frames:
                step = (n - 10) // config.max_frames + 1
                frames = frames[5:n-5:step]

            # Padding (on GPU)
            n = frames.shape[0]
            if n < config.max_frames:
                pad = frames[-1].repeat(config.max_frames - n, 1, 1, 1).to(device)
                frames = torch.cat((frames, pad), dim=0)

            # Close reader
            if config.video_processing_tool == "OpenCV":
                reader.release()
            elif config.video_processing_tool == "VidGear":
                reader.stop()

            return frames, encoding.reshape(enc_len, 1), video_path







# ----------------------------------------------------------
# Dataloader creator
# ----------------------------------------------------------
def get_dataloader(df, phase, batch_size, features_dir=None):
    # Filter out classes with fewer than 2 samples
    class_counts = df['idd'].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df['idd'].isin(valid_classes)]

    # Perform train-test split
    if len(valid_classes) > 1:
        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=config.seed, stratify=df['idd']
        )
    else:
        # Fallback to random splitting if stratified splitting is not possible
        train_df, val_df = train_test_split(
            df, test_size=0.1, random_state=config.seed
        )

    if phase == "train":
        dataset = SLDataset(train_df, features_dir=features_dir)
    else:
        dataset = SLDataset(val_df, features_dir=features_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=pad_collate
    )
    return loader
