import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Only show ERROR logs, hide INFO/DEBUG

import pandas as pd
import os
import torch

from config import config
from data.vocab import encode_sentence
from utils.file_ops import safe_mkdir


def load_sentences(csv_path, limit=None):
    df = pd.read_csv(
        csv_path,
        sep=';',
        encoding='utf-8',
        header=None,
        names=['idd', 'sentence', 'sign_language']
    )
    if limit:
        df = df.iloc[:limit]
    return df


import concurrent.futures

def build_video_table(sentences_df, encodings):
    rows = []
    
    def process_row(row):
        sid = int(row['idd'])
        raw_sentence = str(row['sign_language']).lower()
        encoded = encode_sentence(raw_sentence, encodings)

        dir_path = os.path.join(
            config.video_folder, config.camera_source, str(sid)
        )

        if not os.path.exists(dir_path):
            print("⚠ Missing:", dir_path)
            return []

        file_paths = []
        for f in os.listdir(dir_path):
            fp = os.path.join(dir_path, f)
            if os.path.isfile(fp):
                file_paths.append([sid, fp, encoded])
        return file_paths

    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_row, [row for _, row in sentences_df.iterrows()]))

    # Flatten the list of lists
    rows = [item for sublist in results for item in sublist]

    return pd.DataFrame(rows, columns=["idd", "video_file", "encoding"])
