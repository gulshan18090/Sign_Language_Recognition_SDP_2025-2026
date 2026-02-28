"""
SLRVideoDataset: reads video files, applies augmentation, encodes targets.
Missing idd folders are skipped with a printed warning.
"""
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from config.config import Config
from data.augmentation import VideoAugmentor
from data.video_io import find_video_file, read_video_frames
from utils.vocabulary import Vocabulary


class SLRVideoDataset(Dataset):

    def __init__(
        self,
        records: List[dict],
        vocab: Vocabulary,
        cfg: Config,
        is_train: bool = True,
    ):
        self.vocab       = vocab
        self.cfg         = cfg
        self.is_train    = is_train
        self.videos_root = Path(cfg.paths.videos_root)

        # Filter out records whose idd folder does not exist
        self.records = self._filter_records(records)

        self.augmentor = VideoAugmentor(
            frame_size=tuple(cfg.video.frame_size),
            mean=tuple(cfg.video.mean),
            std=tuple(cfg.video.std),
            random_crop=cfg.augment.random_crop,
            random_crop_scale=tuple(cfg.augment.random_crop_scale),
            horizontal_flip_p=cfg.augment.horizontal_flip_p,
            color_jitter=cfg.augment.color_jitter,
            color_jitter_brightness=cfg.augment.color_jitter_brightness,
            color_jitter_contrast=cfg.augment.color_jitter_contrast,
            color_jitter_saturation=cfg.augment.color_jitter_saturation,
            color_jitter_hue=cfg.augment.color_jitter_hue,
            color_jitter_p=cfg.augment.color_jitter_p,
            grayscale_p=cfg.augment.grayscale_p,
            gaussian_blur_p=cfg.augment.gaussian_blur_p,
            gaussian_blur_kernel=cfg.augment.gaussian_blur_kernel,
            frame_drop_p=cfg.augment.frame_drop_p,
            temporal_reverse_p=cfg.augment.temporal_reverse_p,
            cutout_p=cfg.augment.cutout_p,
            cutout_size=cfg.augment.cutout_size,
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec   = self.records[idx]
        idd   = str(rec[self.cfg.data.id_column])
        gloss = rec[self.cfg.data.target_column]

        frames   = self._load_frames(idd)
        frames   = self.augmentor(frames, is_train=self.is_train)
        src_mask = torch.zeros(frames.shape[0], dtype=torch.bool)
        tgt_ids  = torch.tensor(self.vocab.encode(gloss), dtype=torch.long)

        return {
            "frames":   frames,
            "src_mask": src_mask,
            "tgt_ids":  tgt_ids,
            "idd":      idd,
            "gloss":    gloss,
        }

    # ── Internal ─────────────────────────────────────────────────────────────

    def _filter_records(self, records: List[dict]) -> List[dict]:
        """Remove records whose video folder is missing, with a printed warning."""
        valid, skipped = [], []
        for rec in records:
            idd    = str(rec[self.cfg.data.id_column])
            folder = self.videos_root / idd
            if not folder.exists():
                skipped.append(idd)
            else:
                # Also check at least one valid video file exists
                has_video = any(
                    list(folder.glob(f"*{ext}"))
                    for ext in self.cfg.video.video_extensions
                )
                if not has_video:
                    skipped.append(idd)
                    print(f"[Dataset] No video file found in {folder} — skipping.")
                else:
                    valid.append(rec)

        if skipped:
            print(f"[Dataset] Skipped {len(skipped)} missing folders: {skipped[:10]}"
                  f"{'...' if len(skipped) > 10 else ''}")
        print(f"[Dataset] Loaded {len(valid)} valid samples "
              f"({'train' if self.is_train else 'val/test'})")
        return valid

    def _load_frames(self, idd: str) -> torch.Tensor:
        folder     = self.videos_root / idd
        video_path = find_video_file(folder, self.cfg.video.video_extensions)
        return read_video_frames(
            video_path=str(video_path),
            num_frames=self.cfg.video.num_frames,
            frame_size=tuple(self.cfg.video.frame_size),
            temporal_jitter=self.is_train and self.cfg.augment.temporal_jitter,
        )