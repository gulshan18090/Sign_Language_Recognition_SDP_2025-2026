"""
Inference module: accepts a video file (or folder), returns a gloss string.
Supports greedy and beam search decoding.

Usage:
    predictor = SLRPredictor.from_checkpoint("artifacts/checkpoints/best.pt")
    gloss = predictor.predict("drive/videos_slr/42/sign.mp4")
    print(gloss)   # e.g. "men mektebe getmek"
"""
from pathlib import Path
from typing import List, Optional

import torch

from config.config import Config, get_config
from data.augmentation import VideoAugmentor
from data.video_io import find_video_file, read_video_frames
from model.seq2seq import SLRSeq2Seq
from utils.vocabulary import Vocabulary


class SLRPredictor:

    def __init__(self, model: SLRSeq2Seq, vocab: Vocabulary, cfg: Config):
        self.model  = model
        self.vocab  = vocab
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Inference-time augmentor (no spatial/temporal augmentation, just normalise)
        self.augmentor = VideoAugmentor(
            frame_size=tuple(cfg.video.frame_size),
            mean=tuple(cfg.video.mean),
            std=tuple(cfg.video.std),
            random_crop=False,
            horizontal_flip_p=0.0,
            color_jitter=False,
            grayscale_p=0.0,
            gaussian_blur_p=0.0,
            frame_drop_p=0.0,
            temporal_reverse_p=0.0,
            cutout_p=0.0,
        )

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        vocab_path: Optional[str] = None,
        cfg: Optional[Config] = None,
    ) -> "SLRPredictor":
        cfg = cfg or get_config()
        vocab_path = vocab_path or cfg.paths.vocab_path

        vocab = Vocabulary(
            pad=cfg.vocab.pad_token,
            sos=cfg.vocab.sos_token,
            eos=cfg.vocab.eos_token,
            unk=cfg.vocab.unk_token,
        )
        vocab.load(vocab_path)

        model = SLRSeq2Seq(
            vocab_size=len(vocab),
            vit_model_name=cfg.vit.model_name,
            vit_pretrained=False,     # weights come from checkpoint
            vit_output_dim=cfg.vit.output_dim,
            encoder_d_model=cfg.model.encoder_d_model,
            encoder_nhead=cfg.model.encoder_nhead,
            encoder_num_layers=cfg.model.encoder_num_layers,
            encoder_dim_feedforward=cfg.model.encoder_dim_feedforward,
            encoder_dropout=cfg.model.encoder_dropout,
            decoder_d_model=cfg.model.decoder_d_model,
            decoder_nhead=cfg.model.decoder_nhead,
            decoder_num_layers=cfg.model.decoder_num_layers,
            decoder_dim_feedforward=cfg.model.decoder_dim_feedforward,
            decoder_dropout=cfg.model.decoder_dropout,
            pad_idx=vocab.pad_idx,
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        print(f"[Predictor] Loaded checkpoint  epoch={state.get('epoch','?')}  "
              f"val_loss={state.get('val_loss', 0):.4f}")
        return cls(model=model, vocab=vocab, cfg=cfg)

    # ── Main prediction API ──────────────────────────────────────────────────

    def predict(self, video_path: str, beam_size: int = 4) -> str:
        """
        Predict from a single video file.

        Args:
            video_path : path to .mp4 / .avi / etc.
            beam_size  : 1 = greedy, >1 = beam search
        Returns:
            gloss string, e.g. "men mektebe getmek"
        """
        frames, src_mask = self._load_video(video_path)
        return self._decode(frames, src_mask, beam_size)

    def predict_folder(self, folder_path: str, beam_size: int = 4) -> str:
        """
        Predict from an idd folder (finds first video file inside).
        """
        folder = Path(folder_path)
        video_path = find_video_file(folder, self.cfg.video.video_extensions)
        return self.predict(str(video_path), beam_size=beam_size)

    # ── Decoding ─────────────────────────────────────────────────────────────

    def _decode(self, frames: torch.Tensor, src_mask: torch.Tensor, beam_size: int) -> str:
        if beam_size == 1:
            ids = self._greedy(frames, src_mask)
        else:
            ids = self._beam_search(frames, src_mask, beam_size)
        return self.vocab.decode(ids)

    @torch.no_grad()
    def _greedy(self, frames, src_mask) -> List[int]:
        memory  = self.model.encode(frames, src_mask)
        sos, eos = self.vocab.sos_idx, self.vocab.eos_idx
        generated = [sos]
        for _ in range(self.cfg.model.max_target_len):
            tgt = torch.tensor([generated], dtype=torch.long, device=self.device)
            T   = tgt.size(1)
            cm  = self.model.decoder.make_causal_mask(T, self.device)
            logits = self.model.decoder(tgt=tgt, memory=memory, tgt_mask=cm,
                                        memory_key_padding_mask=src_mask)
            nxt = logits[0, -1].argmax(-1).item()
            generated.append(nxt)
            if nxt == eos:
                break
        return generated

    @torch.no_grad()
    def _beam_search(self, frames, src_mask, beam_size: int = 4) -> List[int]:
        memory  = self.model.encode(frames, src_mask)
        sos, eos = self.vocab.sos_idx, self.vocab.eos_idx
        beams, completed = [(0.0, [sos])], []

        for _ in range(self.cfg.model.max_target_len):
            new_beams = []
            for lp, ids in beams:
                if ids[-1] == eos:
                    completed.append((lp, ids))
                    continue
                tgt = torch.tensor([ids], dtype=torch.long, device=self.device)
                cm  = self.model.decoder.make_causal_mask(tgt.size(1), self.device)
                logits = self.model.decoder(tgt=tgt, memory=memory, tgt_mask=cm,
                                            memory_key_padding_mask=src_mask)
                log_probs = torch.log_softmax(logits[0, -1], dim=-1)
                top_lp, top_tok = log_probs.topk(beam_size)
                for tlp, tok in zip(top_lp.tolist(), top_tok.tolist()):
                    new_beams.append((lp + tlp, ids + [tok]))
            if not new_beams:
                break
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_size]

        completed += beams
        completed.sort(key=lambda x: x[0], reverse=True)
        return completed[0][1]

    # ── Video loading ────────────────────────────────────────────────────────

    def _load_video(self, video_path: str):
        frames = read_video_frames(
            video_path=video_path,
            num_frames=self.cfg.video.num_frames,
            frame_size=tuple(self.cfg.video.frame_size),
            temporal_jitter=False,
        )                                                  # (T, 3, H, W)
        frames = self.augmentor(frames, is_train=False)    # normalised
        src_mask = torch.zeros(self.cfg.video.num_frames, dtype=torch.bool)

        frames   = frames.unsqueeze(0).to(self.device)    # (1, T, 3, H, W)
        src_mask = src_mask.unsqueeze(0).to(self.device)  # (1, T)
        return frames, src_mask
