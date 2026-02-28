"""
Word-level vocabulary built from the sign_language column.
Handles encode / decode and persistence to JSON.
"""
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd


class Vocabulary:
    def __init__(self, pad="<pad>", sos="<sos>", eos="<eos>", unk="<unk>"):
        self.PAD = pad
        self.SOS = sos
        self.EOS = eos
        self.UNK = unk

        self.word2idx = {}
        self.idx2word = {}
        self._built = False

    # ── Build ────────────────────────────────────────────────────────────────

    def build_from_sequences(self, sequences: List[str], min_freq: int = 1) -> None:
        """Build vocab from list of gloss strings."""
        counter = Counter()
        for seq in sequences:
            for tok in self._tokenize(seq):
                counter[tok] += 1

        # Special tokens first (fixed indices)
        specials = [self.PAD, self.SOS, self.EOS, self.UNK]
        self.word2idx = {t: i for i, t in enumerate(specials)}

        for word, freq in sorted(counter.items()):
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx

        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self._built = True

    # ── Encode / Decode ──────────────────────────────────────────────────────

    def encode(self, sequence: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        tokens = self._tokenize(sequence)
        ids = [self.word2idx.get(t, self.word2idx[self.UNK]) for t in tokens]
        if add_sos:
            ids = [self.word2idx[self.SOS]] + ids
        if add_eos:
            ids = ids + [self.word2idx[self.EOS]]
        return ids

    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        words = [self.idx2word.get(i, self.UNK) for i in ids]
        if strip_special:
            specials = {self.PAD, self.SOS, self.EOS}
            words = [w for w in words if w not in specials]
            # stop at first UNK only if desired – keep for transparency
        return " ".join(words)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word2idx, f, ensure_ascii=False, indent=2)
        print(f"[Vocabulary] Saved {len(self)} tokens → {path}")

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.word2idx = json.load(f)
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self._built = True
        print(f"[Vocabulary] Loaded {len(self)} tokens from {path}")

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenizer; lowercases."""
        return text.lower().split()

    def __len__(self):
        return len(self.word2idx)

    @property
    def pad_idx(self): return self.word2idx[self.PAD]

    @property
    def sos_idx(self): return self.word2idx[self.SOS]

    @property
    def eos_idx(self): return self.word2idx[self.EOS]

    @property
    def unk_idx(self): return self.word2idx[self.UNK]
