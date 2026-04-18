"""
Central configuration for Azerbaijani Sign Language Recognition Pipeline.
VIDEO + ViT edition.
"""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PathConfig:
    csv_path: str = "/home/temporaryuser2/Desktop/sdp_module/lstm/drive/sentences_all.csv"
    videos_root: str = "/home/temporaryuser2/Desktop/sdp_module/lstm/drive/Video/Cam2"        # <idd>/<clip>.mp4  (or .avi)
    vocab_path: str = "artifacts/vocab.json"
    checkpoint_dir: str = "artifacts/checkpoints"
    log_dir: str = "artifacts/logs"


@dataclass
class VideoConfig:
    """Controls how raw video is read and sampled into frames."""
    num_frames: int = 16            # frames uniformly sampled per clip
    frame_size: Tuple[int, int] = (224, 224)   # H x W fed to ViT
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406)   # ImageNet stats
    std:  Tuple[float, ...] = (0.229, 0.224, 0.225)
    # Supported extensions (searched in <videos_root>/<idd>/)
    video_extensions: List[str] = field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])


@dataclass
class AugmentConfig:
    """Spatial + temporal augmentations applied during training only."""
    # Spatial
    random_crop: bool = True          # random crop then resize to frame_size
    random_crop_scale: Tuple[float, float] = (0.7, 1.0)
    horizontal_flip_p: float = 0.5
    color_jitter: bool = True
    color_jitter_brightness: float = 0.3
    color_jitter_contrast: float = 0.3
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    color_jitter_p: float = 0.8       # probability of applying color jitter
    grayscale_p: float = 0.1
    gaussian_blur_p: float = 0.3
    gaussian_blur_kernel: int = 5
    # Temporal
    temporal_jitter: bool = True      # randomly shift the sampling grid
    frame_drop_p: float = 0.1         # randomly zero-out each frame with this prob
    temporal_reverse_p: float = 0.1  # reverse frame order (mirror in time)
    # Regularisation
    cutout_p: float = 0.3             # random square patch → zero (CutOut)
    cutout_size: float = 0.15         # patch side = cutout_size * frame_size


@dataclass
class DataConfig:
    csv_sep: str = ";"
    csv_columns: List[str] = field(default_factory=lambda: ["idd", "sentence", "sign_language"])
    target_column: str = "sign_language"
    id_column: str = "idd"
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    seed: int = 42
    batch_size: int = 8               # lower than feature pipeline — video frames are heavier
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class VocabConfig:
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"
    min_freq: int = 1


@dataclass
class ViTConfig:
    """ViT frame encoder settings."""
    model_name: str = "vit_small_patch16_224"  # timm model name (pretrained on ImageNet)
    pretrained: bool = True
    freeze_backbone: bool = False     # set True for first N epochs to warm-up decoder
    freeze_epochs: int = 5            # unfreeze ViT after this many epochs
    output_dim: int = 384             # ViT-Small output dim; adjust per model_name


@dataclass
class ModelConfig:
    # Temporal Transformer Encoder on top of ViT frame features
    vit_output_dim: int = 384         # must match ViTConfig.output_dim
    encoder_d_model: int = 256
    encoder_nhead: int = 8
    encoder_num_layers: int = 4
    encoder_dim_feedforward: int = 1024
    encoder_dropout: float = 0.1
    # Decoder
    decoder_d_model: int = 256
    decoder_nhead: int = 8
    decoder_num_layers: int = 4
    decoder_dim_feedforward: int = 1024
    decoder_dropout: float = 0.1
    max_target_len: int = 30
    label_smoothing: float = 0.1


@dataclass
class TrainConfig:
    epochs: int = 200
    # Two LR groups: backbone gets smaller LR
    lr_backbone: float = 1e-5
    lr_head: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 300
    clip_grad_norm: float = 1.0
    patience: int = 200
    save_top_k: int = 3
    mixed_precision: bool = True


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    vit: ViTConfig = field(default_factory=ViTConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


_cfg = None

def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = Config()
    return _cfg
