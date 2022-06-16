"""
Demonstration program
"""
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.utils.audio import generate_segmented_data
from src.utils.preprocess import FeatureExtractor, expand_mel_spec

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default="feat_ext")
parser.add_argument("--audio")
parser.add_argument("--model")
parser.add_argument("--label", default="COVID-19")
args = parser.parse_args()

# model_path_dict = {
#     "baseline": "dumps/models/baseline_model",
#     "weight_init": "dumps/models/best_exp_weight",
#     "feat_ext": "dumps/models/best_exp_feat",
# }

offset_dict = {"baseline": 13315, "tl": 23680}

# Load the audio data
audio_data, _ = librosa.load(args.audio, sr=16000)
audio_segments, final_labels = generate_segmented_data(
    samples_data=audio_data.reshape((1, -1)), audio_labels=np.array([[args.label]])
)

feature_extractor = FeatureExtractor(
    backup_every_stage=False, offset=offset_dict[args.mode], for_training=False
)
X_test, y_test = feature_extractor.run(
    aggregated_data=audio_segments,
    aggregated_labels=final_labels,
    n_mels=64,
    hop_length=128,
)

# Expand dimension because ResNet expects 3D shape
X_test_expanded = expand_mel_spec(X_test)

# Load model based on what mode this demo is
model = tf.keras.models.load_model(args.model)

# Do prediction
y_proba = model.predict(X_test_expanded)

# Make label to 0 and 1
y_pred = np.where(y_proba >= 0.5, 1, 0)

print(f"There are {len(X_test)} segments of cough.")

for idx, (pred_res, true_res) in enumerate(zip(y_pred, y_test)):
    print(f"Result for segment no. {idx}")
    # Show the spectrogram
    fig, ax = plt.subplots()
    img = librosa.display.specshow(
        data=X_test[idx], sr=16000, x_axis="time", y_axis="mel", fmax=8000, ax=ax
    )
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title(f"Log mel spectrogram for cough segment no. {idx}")
    plt.show()

    # Print the true result and actual result
    print(f"True result: {true_res}")
    print(f"Actual result: {'COVID-19' if pred_res else 'healthy'}")
