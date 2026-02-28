"""
Run full evaluation on the test set and report WER / Seq-Acc / BLEU-4.

Usage:
    python evaluate.py --checkpoint artifacts/checkpoints/best.pt
"""
import argparse
import torch
from config.config import get_config
from data.datamodule import SLRDataModule
from inference.predictor import SLRPredictor
from training.evaluate import evaluate_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    parser.add_argument("--beam_size",  type=int, default=4)
    args = parser.parse_args()

    cfg       = get_config()
    predictor = SLRPredictor.from_checkpoint(args.checkpoint, cfg=cfg)

    # Rebuild test split
    dm = SLRDataModule(cfg)
    dm.setup()
    test_dl = dm.test_dataloader()

    predictions, references = [], []
    device = predictor.device

    for batch in test_dl:
        features = batch["features"].to(device)
        src_mask = batch["src_mask"].to(device)
        glosses  = batch["gloss"]      # list of ground-truth strings

        memory = predictor.model.encode(features, src_mask)

        for i, ref_gloss in enumerate(glosses):
            mem_i  = memory[i:i+1]
            mask_i = src_mask[i:i+1]
            if args.beam_size == 1:
                ids = predictor._greedy_decode(features[i:i+1], src_mask[i:i+1])
            else:
                ids = predictor._beam_search(features[i:i+1], src_mask[i:i+1], args.beam_size)
            pred = predictor.vocab.decode(ids)
            predictions.append(pred)
            references.append(ref_gloss)

    metrics = evaluate_batch(predictions, references)
    print("\n=== Test Set Results ===")
    print(f"  WER         : {metrics['wer']:.4f}  (lower is better)")
    print(f"  Seq Accuracy: {metrics['seq_acc']:.4f}  (higher is better)")
    print(f"  BLEU-4      : {metrics['bleu4']:.4f}  (higher is better)")
    print(f"  Samples     : {metrics['n_samples']}")


if __name__ == "__main__":
    main()
