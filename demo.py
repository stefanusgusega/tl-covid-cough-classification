"""
Demonstration program
"""
import argparse
import librosa
import numpy as np

from src.utils.audio import generate_segmented_data

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="best_exp_feat")
parser.add_argument("--audio")
parser.add_argument("--label", default="COVID-19")
args = parser.parse_args()

model_path_dict = {
    "baseline": "dumps/models/baseline_model",
    "weight_init": "dumps/models/best_exp_weight",
    "feat_ext": "dumps/models/best_exp_feat",
}

# Load the audio data
audio_data, _ = librosa.load(args.audio, sr=16000)
audio_segments, final_labels = generate_segmented_data(
    samples_data=audio_data.reshape((-1, 1)), audio_labels=np.array([[args.label]])
)

# TODO : add feature extractor class
