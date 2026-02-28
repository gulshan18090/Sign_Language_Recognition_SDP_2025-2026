"""
Inference script for Sign Language Recognition
Takes a video path and outputs predicted sentence
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import torch
import torch.nn.functional as F
import argparse
import warnings
from typing import List, Tuple, Dict
warnings.filterwarnings("ignore")

from config import Config as config
from data.vocab import build_word_dict
from data.dataset_builder import load_sentences
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from extract_features import extract_features_from_video

def load_models(encodings, device):
    """Load trained encoder and decoder models"""
    input_size = 512  # Pre-extracted feature dimension (256*2 from bidirectional)
    hidden_size = 256  # Must match the hidden_size used during training
    
    # Initialize models with same architecture as training
    # use_cnn=False, biDirectional=True to match feature extraction
    encoder = EncoderRNN(input_size, hidden_size, device=device, biDirectional=True, use_cnn=False).to(device)
    encoder_output_size = hidden_size * 2  # 256 * 2 = 512 for bidirectional
    decoder = AttnDecoderRNN(
        encoder_output_size, 
        len(encodings), 
        device=device, 
        encoder_hidden_size=encoder_output_size
    ).to(device)
    
    # Load trained weights
    encoder_path = f"{config.drive_folder}/jamal/encoder.model"
    decoder_path = f"{config.drive_folder}/jamal/decoder.model"
    
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder model not found: {encoder_path}")
    if not os.path.exists(decoder_path):
        raise FileNotFoundError(f"Decoder model not found: {decoder_path}")
    
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    
    encoder.eval()
    decoder.eval()
    
    print(f"✅ Loaded models from:\n  {encoder_path}\n  {decoder_path}")
    
    return encoder, decoder

def get_video_features(video_path, features_dir="features"):
    """Get features for a video, either from cache or extract on-the-fly"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    feature_file = os.path.join(features_dir, f"{video_name}.pt")
    
    if os.path.exists(feature_file):
        print(f"📂 Loading cached features: {feature_file}")
        features = torch.load(feature_file, weights_only=True)
    else:
        print(f"🔄 Extracting features from video (this may take a moment)...")
        # Extract features on-the-fly
        extract_features_from_video(video_path, feature_file)
        features = torch.load(feature_file, weights_only=True)
    
    return features

def preprocess_features(features, device):
    """Preprocess features to match training format"""
    # features shape: (T, 256)
    # Need to pad/truncate to max_frames
    T = features.shape[0]
    target_length = config.max_frames
    
    if T < target_length:
        # Pad
        pad_len = target_length - T
        pad = torch.zeros((pad_len, features.shape[1]), dtype=features.dtype)
        features = torch.cat([features, pad], dim=0)
    elif T > target_length:
        # Truncate (take first max_frames)
        features = features[:target_length]
    
    # Add batch dimension: (1, T, 256)
    features = features.unsqueeze(0).to(device)
    
    return features

# Add debug prints for QA

def decode_sequence(encoder, decoder, features, encodings, reverse_encodings, device, max_length=20):
    """Decode video features to sentence using greedy decoding (argmax)"""
    with torch.no_grad():
        # Initialize encoder hidden state
        batch_size = features.shape[0]
        encoder_hidden = encoder.initHidden(batch_size)

        # Encode video features
        encoder_output, encoder_hidden = encoder(features, encoder_hidden)

        # Prepare decoder hidden state (merge bidirectional if needed)
        if encoder.D == 2:
            h = torch.cat((encoder_hidden[0][0:1], encoder_hidden[0][1:2]), dim=2)
            c = torch.cat((encoder_hidden[1][0:1], encoder_hidden[1][1:2]), dim=2)
            decoder_hidden = (h, c)
        else:
            decoder_hidden = encoder_hidden

        # Start with SOS token
        sos_token = encodings.get('SOS', 0)
        if sos_token not in encodings.values():
            # If SOS not found, use first token (index 0)
            sos_token = 0
        decoder_input = torch.tensor([[sos_token]], device=device)

        decoded_words = []
        decoded_token_ids = []
        eos_token = encodings.get('EOS', 1)

        # Decode sequence with greedy decoding (argmax)
        for step in range(max_length):
            output, decoder_hidden, attn_weights = decoder(
                decoder_input, decoder_hidden, encoder_output
            )

            # Get predicted token (argmax - most likely token)
            token_id = output.argmax(dim=1).item()
            decoded_token_ids.append(token_id)

            # Check for EOS token - stop generation
            if token_id == eos_token:
                break

            # Skip SOS token if it appears in the middle
            if token_id == sos_token and step > 0:
                continue

            # Convert token to word
            word = reverse_encodings.get(token_id, '<UNK>')

            # Skip special tokens and unknown
            if word in ['<UNK>', 'SOS']:
                continue

            decoded_words.append(word)

            # Use predicted token as next input
            decoder_input = torch.tensor([[token_id]], device=device)

        print('Decoded token IDs:', decoded_token_ids)
        print('Decoded words:', decoded_words)
        return decoded_words


def _token_ids_to_words(
    token_ids: List[int],
    encodings: Dict[str, int],
    reverse_encodings: Dict[int, str]
) -> List[str]:
    """Convert predicted token ids to clean word list."""
    sos_token = encodings.get('SOS', 0)
    eos_token = encodings.get('EOS', 1)

    words = []
    for idx, token_id in enumerate(token_ids):
        if token_id == eos_token:
            break
        if token_id == sos_token and idx > 0:
            continue
        word = reverse_encodings.get(token_id, '<UNK>')
        if word in ['<UNK>', 'SOS', 'EOS']:
            continue
        words.append(word)
    return words


def decode_sequence_beam(
    encoder,
    decoder,
    features,
    encodings,
    reverse_encodings,
    device,
    beam_width=5,
    max_length=20,
    top_k=5
) -> List[Tuple[str, float]]:
    """
    Beam-search decoding that returns n-best sentence hypotheses.
    Returns a list of tuples: (sentence, log_prob), sorted by log_prob desc.
    """
    if beam_width < 1:
        beam_width = 1
    if top_k < 1:
        top_k = 1

    with torch.no_grad():
        batch_size = features.shape[0]
        encoder_hidden = encoder.initHidden(batch_size)
        encoder_output, encoder_hidden = encoder(features, encoder_hidden)

        if encoder.D == 2:
            h = torch.cat((encoder_hidden[0][0:1], encoder_hidden[0][1:2]), dim=2)
            c = torch.cat((encoder_hidden[1][0:1], encoder_hidden[1][1:2]), dim=2)
            init_hidden = (h, c)
        else:
            init_hidden = encoder_hidden

        sos_token = encodings.get('SOS', 0)
        eos_token = encodings.get('EOS', 1)

        beams = [{
            'tokens': [sos_token],
            'score': 0.0,
            'hidden': init_hidden,
            'done': False
        }]
        completed = []

        for _ in range(max_length):
            expanded = []
            all_done = True

            for beam in beams:
                if beam['done']:
                    expanded.append(beam)
                    continue

                all_done = False
                decoder_input = torch.tensor([[beam['tokens'][-1]]], device=device)
                output, next_hidden, _ = decoder(decoder_input, beam['hidden'], encoder_output)
                log_probs = F.log_softmax(output, dim=1).squeeze(0)
                top_log_probs, top_indices = torch.topk(log_probs, k=min(beam_width, log_probs.shape[0]))

                for lp, idx in zip(top_log_probs.tolist(), top_indices.tolist()):
                    new_tokens = beam['tokens'] + [idx]
                    is_done = (idx == eos_token)
                    new_beam = {
                        'tokens': new_tokens,
                        'score': beam['score'] + lp,
                        'hidden': next_hidden,
                        'done': is_done
                    }
                    expanded.append(new_beam)
                    if is_done:
                        completed.append(new_beam)

            if all_done:
                break

            expanded.sort(key=lambda x: x['score'], reverse=True)
            beams = expanded[:beam_width]

        if not completed:
            completed = beams

        completed.sort(key=lambda x: x['score'], reverse=True)
        seen = set()
        nbest = []
        for cand in completed:
            words = _token_ids_to_words(cand['tokens'][1:], encodings, reverse_encodings)
            sentence = ' '.join(words).strip()
            if sentence in seen:
                continue
            seen.add(sentence)
            nbest.append((sentence, cand['score']))
            if len(nbest) >= top_k:
                break

        return nbest

def predict_video(video_path, encodings, reverse_encodings, encoder, decoder, device):
    """Main prediction function"""
    print(f"\n🎥 Processing video: {video_path}")
    
    # Get features
    features = get_video_features(video_path)
    
    # Preprocess features
    features = preprocess_features(features, device)
    
    # Decode to sentence
    decoded_words = decode_sequence(encoder, decoder, features, encodings, reverse_encodings, device)
    
    # Join words into sentence
    sentence = ' '.join(decoded_words)
    
    return sentence

def main():
    parser = argparse.ArgumentParser(description="SLR Inference - Predict sentence from video")
    parser.add_argument("video_path", type=str, help="Path to input video file")
    parser.add_argument("--features-dir", type=str, default="features", 
                       help="Directory containing pre-extracted features")
    parser.add_argument("--beam-width", type=int, default=1,
                       help="Beam width for decoding (1 = greedy)")
    parser.add_argument("--top-k", type=int, default=5,
                       help="Number of hypotheses to print when beam-width > 1")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return

    device = torch.device(config.device)
    print(f"🖥️  Using device: {device}")

    # Load vocabulary (same as training - must match training vocab size)
    print("\n📚 Loading vocabulary...")
    # IMPORTANT: Must use same limit as training (400) to match vocabulary size
    df_sent = load_sentences(config.train_csv_path, limit=400)
    encodings, reverse_encodings = build_word_dict(df_sent["sign_language"])
    print(f"   Vocabulary size: {len(encodings)}")
    print('Encodings sample:', list(encodings.items())[:10])
    print('Reverse encodings sample:', list(reverse_encodings.items())[:10])

    # Load models
    print("\n🤖 Loading trained models...")
    encoder, decoder = load_models(encodings, device)
    print('Encoder:', encoder)
    print('Decoder:', decoder)

    # Predict
    print("\n🔮 Running inference...")
    features = get_video_features(args.video_path)
    print('Feature shape:', features.shape)
    print('Feature sample:', features.flatten()[:10])
    features = preprocess_features(features, device)
    if args.beam_width > 1:
        hypotheses = decode_sequence_beam(
            encoder, decoder, features, encodings, reverse_encodings, device,
            beam_width=args.beam_width, top_k=args.top_k
        )
        predicted_sentence = hypotheses[0][0].split() if hypotheses else []
        print("\n🔎 TOP HYPOTHESES:")
        for rank, (sent, score) in enumerate(hypotheses, start=1):
            print(f"   {rank:>2d}. [{score:.3f}] {sent}")
    else:
        predicted_sentence = decode_sequence(
            encoder, decoder, features, encodings, reverse_encodings, device
        )

    # Output result
    print("\n" + "="*60)
    print("📝 PREDICTED SENTENCE:")
    print("="*60)
    print(f"   {' '.join(predicted_sentence)}")
    print("="*60 + "\n")

    return predicted_sentence

if __name__ == "__main__":
    main()
