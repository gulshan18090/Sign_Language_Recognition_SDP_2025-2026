"""
DataModule: reads CSV, builds vocab, creates train/val/test DataLoaders.
"""
import random
from functools import partial
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from config.config import Config
from data.collate import slr_collate_fn
from data.dataset import SLRVideoDataset
from utils.vocabulary import Vocabulary


class SLRDataModule:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.vocab = Vocabulary(
            pad=cfg.vocab.pad_token,
            sos=cfg.vocab.sos_token,
            eos=cfg.vocab.eos_token,
            unk=cfg.vocab.unk_token,
        )
        self._train_dl = None
        self._val_dl   = None
        self._test_dl  = None

    def setup(self) -> None:
        records = self._load_csv()
        self._build_vocab(records)

        random.seed(self.cfg.data.seed)
        random.shuffle(records)

        n       = len(records)
        n_train = int(n * self.cfg.data.train_ratio)
        n_val   = int(n * self.cfg.data.val_ratio)

        train_recs = records[:n_train]
        val_recs   = records[n_train : n_train + n_val]
        test_recs  = records[n_train + n_val :]

        print(f"[DataModule] Split  train={len(train_recs)}  val={len(val_recs)}  test={len(test_recs)}")

        collate = partial(slr_collate_fn, pad_idx=self.vocab.pad_idx)
        self._train_dl = self._make_loader(train_recs, is_train=True,  collate=collate)
        self._val_dl   = self._make_loader(val_recs,   is_train=False, collate=collate)
        self._test_dl  = self._make_loader(test_recs,  is_train=False, collate=collate)

    def train_dataloader(self): return self._train_dl
    def val_dataloader(self):   return self._val_dl
    def test_dataloader(self):  return self._test_dl

    def _load_csv(self):
        df = pd.read_csv(
            self.cfg.paths.csv_path,
            sep=self.cfg.data.csv_sep,
            header=None,
            names=self.cfg.data.csv_columns,
        )
        df = df.dropna(subset=[self.cfg.data.target_column, self.cfg.data.id_column])
        print(f"[DataModule] Loaded {len(df)} rows from {self.cfg.paths.csv_path}")
        return df.to_dict("records")

    def _build_vocab(self, records) -> None:
        vocab_path = Path(self.cfg.paths.vocab_path)
        if vocab_path.exists():
            self.vocab.load(str(vocab_path))
            return
        sequences = [r[self.cfg.data.target_column] for r in records]
        self.vocab.build_from_sequences(sequences, min_freq=self.cfg.vocab.min_freq)
        self.vocab.save(str(vocab_path))
        print(f"[DataModule] Vocabulary size: {len(self.vocab)}")

    def _make_loader(self, records, is_train, collate) -> DataLoader:
        ds = SLRVideoDataset(
            records=records,
            vocab=self.vocab,
            cfg=self.cfg,
            is_train=is_train,
        )
        return DataLoader(
            ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=is_train,
            num_workers=self.cfg.data.num_workers,
            pin_memory=self.cfg.data.pin_memory,
            collate_fn=collate,
        )
