"""
Minimal CLI inference:
    python predict.py --pt drive/features_slr/42/frame_001.pt
    python predict.py --pt /path/to/video.mp4
    python predict.py --folder /path/to/idd_folder/
"""
import argparse
from inference.predictor import SLRPredictor


def main():
    parser = argparse.ArgumentParser(description="Azerbaijani SLR Inference")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pt",     type=str, help="Single input file (.pt feature tensor OR video file)")
    group.add_argument("--folder", type=str, help="Sample folder (first .pt or first video file is used)")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/epoch_100_loss_nan.pt")
    parser.add_argument("--beam_size",  type=int, default=4, help="1=greedy, >1=beam search")
    args = parser.parse_args()

    predictor = SLRPredictor.from_checkpoint(args.checkpoint)

    if args.pt:
        result = predictor.predict(args.pt, beam_size=args.beam_size)
    else:
        result = predictor.predict_folder(args.folder, beam_size=args.beam_size)

    print(f"Predicted gloss: {result}")


if __name__ == "__main__":
    main()
