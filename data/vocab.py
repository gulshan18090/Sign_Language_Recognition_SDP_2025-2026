import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Only show ERROR logs, hide INFO/DEBUG

from config import config

def build_word_dict(sentences_series):
    word_set = set(['SOS', 'EOS'])
    sentences_series.str.lower().str.split().apply(word_set.update)

    sorted_words = sorted(word_set)
    encodings = {w: i for i, w in enumerate(sorted_words)}
    reverse = {i: w for w, i in encodings.items()}
    return encodings, reverse


def encode_sentence(sentence, encodings):
    tokens = ('SOS ' + sentence + ' EOS').split()
    encoded = [encodings.get(tok, 0) for tok in tokens]

    # pad/cut
    if len(encoded) > config.max_words_in_sentence:
        encoded = encoded[:config.max_words_in_sentence]
    else:
        encoded += [0] * (config.max_words_in_sentence - len(encoded))

    return encoded
