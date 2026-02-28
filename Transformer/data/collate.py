"""
Custom collate function for video batches.
frames are all the same shape (T, 3, H, W) so stacking is direct.
tgt_ids are variable length and need padding.
"""
from typing import List, Dict

import torch
from torch.nn.utils.rnn import pad_sequence


def slr_collate_fn(batch: List[Dict], pad_idx: int) -> Dict[str, torch.Tensor]:
    """
    Returns:
        frames    : (B, T, 3, H, W)    raw frames (ViT backbone processes per-frame)
        src_mask  : (B, T)             True = PAD (all False for uniform-sampled clips)
        tgt_in    : (B, T_tgt-1)       decoder input  (<sos> w1 ... wN)
        tgt_out   : (B, T_tgt-1)       expected output (w1 ... wN <eos>)
        tgt_mask  : (B, T_tgt-1)       True = PAD
    """
    frames   = torch.stack([s["frames"]   for s in batch])    # (B, T, 3, H, W)
    src_mask = torch.stack([s["src_mask"] for s in batch])    # (B, T)

    tgt_ids    = [s["tgt_ids"] for s in batch]
    tgt_padded = pad_sequence(tgt_ids, batch_first=True, padding_value=pad_idx)

    tgt_in  = tgt_padded[:, :-1]
    tgt_out = tgt_padded[:, 1:]
    tgt_mask = (tgt_in == pad_idx)

    return {
        "frames":   frames,
        "src_mask": src_mask,
        "tgt_in":   tgt_in,
        "tgt_out":  tgt_out,
        "tgt_mask": tgt_mask,
        "gloss":    [s["gloss"] for s in batch],
    }