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
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return

    device = torch.device(config.device)
    print(f"🖥️  Using device: {device}")

    # Load vocabulary (same as training - must match training vocab size)
    print("\n📚 Loading vocabulary...")
    df_sent = load_sentences(config.train_csv_path, limit=None)  # Load all for full vocab
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

