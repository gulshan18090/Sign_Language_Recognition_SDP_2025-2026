"""
Trainer with:
  - Two LR groups: ViT backbone (small LR) vs temporal encoder+decoder (large LR)
  - Backbone freeze/unfreeze schedule
  - Warmup + cosine LR decay
  - AMP mixed precision
  - Gradient clipping
  - Early stopping
  - Best-k checkpointing
  - TensorBoard logging
"""
import math
import os
import heapq
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from config.config import Config
from model.seq2seq import SLRSeq2Seq


class Trainer:

    def __init__(self, model: SLRSeq2Seq, cfg: Config, pad_idx: int):
        self.model   = model
        self.cfg     = cfg
        self.pad_idx = pad_idx
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=cfg.model.label_smoothing,
        )

        # Two param groups: backbone gets 10x smaller LR
        backbone_params = list(model.frame_encoder.parameters())
        head_params     = (
            list(model.temporal_encoder.parameters()) +
            list(model.decoder.parameters())
        )
        self.optimizer = AdamW([
            {"params": backbone_params, "lr": cfg.train.lr_backbone},
            {"params": head_params,     "lr": cfg.train.lr_head},
        ], weight_decay=cfg.train.weight_decay)

        self.scheduler = self._build_scheduler()
        self.scaler    = GradScaler(enabled=cfg.train.mixed_precision and self.device.type == "cuda")

        Path(cfg.paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        self.writer   = SummaryWriter(log_dir=cfg.paths.log_dir)

        self._ckpt_heap  = []
        self._best_val   = float("inf")
        self._no_improve = 0
        self._global_step = 0

        # Start with frozen backbone if configured
        if cfg.vit.freeze_backbone:
            self.model.freeze_backbone()

    # ── Public API ───────────────────────────────────────────────────────────

    def fit(self, train_dl, val_dl) -> None:
        total = self.model.num_parameters_total
        trainable = self.model.num_parameters
        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Params  total={total:,}  trainable={trainable:,}")

        for epoch in range(1, self.cfg.train.epochs + 1):

            # Unfreeze backbone after warm-up epochs
            if self.cfg.vit.freeze_backbone and epoch == self.cfg.vit.freeze_epochs + 1:
                self.model.unfreeze_backbone()
                print(f"[Trainer] Epoch {epoch}: ViT backbone unfrozen.")

            train_loss = self._train_epoch(train_dl, epoch)
            val_loss   = self._val_epoch(val_dl, epoch)

            self.writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            print(f"Epoch {epoch:03d}  train={train_loss:.4f}  val={val_loss:.4f}  "
                  f"lr_head={self.optimizer.param_groups[1]['lr']:.2e}")

            improved = self._save_checkpoint(epoch, val_loss)
            self._no_improve = 0 if improved else self._no_improve + 1
            if self._no_improve >= self.cfg.train.patience:
                print(f"[Trainer] Early stopping at epoch {epoch}.")
                break

        self.writer.close()
        print(f"[Trainer] Best val_loss: {self._best_val:.4f}")

    # ── Epoch loops ──────────────────────────────────────────────────────────

    def _train_epoch(self, dl, epoch: int) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        for batch in dl:
            frames   = batch["frames"].to(self.device)
            src_mask = batch["src_mask"].to(self.device)
            tgt_in   = batch["tgt_in"].to(self.device)
            tgt_out  = batch["tgt_out"].to(self.device)
            tgt_mask = batch["tgt_mask"].to(self.device)

            self.optimizer.zero_grad()
            use_amp = self.cfg.train.mixed_precision and self.device.type == "cuda"
            with autocast(enabled=use_amp):
                logits = self.model(frames, src_mask, tgt_in, tgt_mask)
                loss   = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_out.reshape(-1),
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            self._global_step += 1
            self.writer.add_scalar("lr/head", self.optimizer.param_groups[1]["lr"], self._global_step)
            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def _val_epoch(self, dl, epoch: int) -> float:
        self.model.eval()
        total_loss, n = 0.0, 0
        for batch in dl:
            frames   = batch["frames"].to(self.device)
            src_mask = batch["src_mask"].to(self.device)
            tgt_in   = batch["tgt_in"].to(self.device)
            tgt_out  = batch["tgt_out"].to(self.device)
            tgt_mask = batch["tgt_mask"].to(self.device)

            logits = self.model(frames, src_mask, tgt_in, tgt_mask)
            loss   = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
            )
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    # ── LR Schedule ──────────────────────────────────────────────────────────

    def _build_scheduler(self) -> LambdaLR:
        warmup = self.cfg.train.warmup_steps
        total  = self.cfg.train.epochs * 100  # rough estimate

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total - warmup))
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return LambdaLR(self.optimizer, lr_lambda)

    # ── Checkpointing ────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float) -> bool:
        ckpt_path = Path(self.cfg.paths.checkpoint_dir) / f"epoch_{epoch:03d}_loss_{val_loss:.4f}.pt"
        state = {
            "epoch": epoch,
            "val_loss": val_loss,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(state, ckpt_path)
        heapq.heappush(self._ckpt_heap, (val_loss, str(ckpt_path)))

        while len(self._ckpt_heap) > self.cfg.train.save_top_k:
            worst_loss, worst_path = heapq.heappop(self._ckpt_heap)
            if worst_path != str(ckpt_path) and Path(worst_path).exists():
                os.remove(worst_path)

        if val_loss < self._best_val:
            self._best_val = val_loss
            torch.save(state, Path(self.cfg.paths.checkpoint_dir) / "best.pt")
            return True
        return False
