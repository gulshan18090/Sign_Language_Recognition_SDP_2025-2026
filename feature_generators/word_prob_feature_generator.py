import torch
import numpy as np

def sentence_to_word_probs(sentence, encodings, vocab_size):
    """
    Converts a sentence to a word probability vector (bag-of-words style).
    Args:
        sentence (str): The sentence to encode.
        encodings (dict): word->index mapping.
        vocab_size (int): Size of the vocabulary.
    Returns:
        np.ndarray: 1D array of word probabilities (length = vocab_size)
    """
    tokens = sentence.lower().split()
    vec = np.zeros(vocab_size, dtype=np.float32)
    for tok in tokens:
        idx = encodings.get(tok)
        if idx is not None:
            vec[idx] += 1
    if vec.sum() > 0:
        vec = vec / vec.sum()  # Normalize to probabilities
    return vec

def generate_word_prob_features(sentences, encodings):
    vocab_size = len(encodings)
    features = [sentence_to_word_probs(s, encodings, vocab_size) for s in sentences]
    return np.stack(features)

# Example usage:
# encodings = torch.load('encodings.dict')
# sentences = ["mən bakı yaşamaq", "mən hansı sənəd vermək"]
# features = generate_word_prob_features(sentences, encodings)
# print(features.shape)  # (num_sentences, vocab_size)
