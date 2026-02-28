"""
Training entry point.

Usage:
    python train.py
"""
from config.config import get_config
from data.datamodule import SLRDataModule
from model.seq2seq import SLRSeq2Seq
from training.trainer import Trainer


def main():
    cfg = get_config()

    dm = SLRDataModule(cfg)
    dm.setup()

    vocab      = dm.vocab
    vocab_size = len(vocab)
    print(f"Vocabulary size : {vocab_size}")

    model = SLRSeq2Seq(
        vocab_size=vocab_size,
        vit_model_name=cfg.vit.model_name,
        vit_pretrained=cfg.vit.pretrained,
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
    print(f"Total params    : {model.num_parameters_total:,}")
    print(f"Trainable params: {model.num_parameters:,}")

    trainer = Trainer(model=model, cfg=cfg, pad_idx=vocab.pad_idx)
    trainer.fit(dm.train_dataloader(), dm.val_dataloader())
    print("Training complete.")


if __name__ == "__main__":
    main()