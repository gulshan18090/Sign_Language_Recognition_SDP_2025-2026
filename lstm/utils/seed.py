
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)  # Only show ERROR logs, hide INFO/DEBUG
import random
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
