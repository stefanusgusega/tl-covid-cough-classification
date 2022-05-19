"""
Main program for testing model
"""
import argparse
import os
import sys
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import tensorflow as tf

# from icecream import ic

from src.utils.preprocess import FeatureExtractor, encode_label, expand_mel_spec
from src.utils.chore import generate_now_datetime, load_obj_from_pkl

DUMP_PATH = "dumps"
LOG_PATH = os.path.join(DUMP_PATH, "logs")
MODEL_DIR = os.path.join(DUMP_PATH, "models")
DATA_PATH = os.path.join(DUMP_PATH, "data")
OFFSET = 23680

sys.stdout = open(
    os.path.join("logs", "testing", f"pretrain_{generate_now_datetime()}.txt"),
    "w",
    encoding="utf-8",
)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-m")
parser.add_argument("-td")
args = parser.parse_args()

MODEL = args.m
DATA_FILE = os.path.join(DATA_PATH, args.td)

# Loading testing dataset
X_test_raw, y_test_raw = load_obj_from_pkl(DATA_FILE)

# Preprocessing dataset
extractor = FeatureExtractor(
    pickle_folder=os.path.join(DATA_PATH, "pretrain-test", "preprocessed"),
    for_training=False,
    offset=OFFSET,
)
X_test, y_test = extractor.run(
    X_test_raw,
    y_test_raw,
    sampling_rate=16000,
    hop_length=128,
    n_mels=64,
)

# Encode the y_test
y_test = encode_label(y_test, pos_label="cough")

# Expand the dimension of the data because ResNet expects 3D
X_test = expand_mel_spec(X_test)

# ic(X_test.shape)
# Load model first
model = tf.keras.models.load_model(os.path.join(MODEL_DIR, MODEL))
# ic(model_paths)

# Do prediction
y_softmax = model.predict(X_test)

# Make label to 0 and 1
y_pred = np.argmax(y_softmax, axis=1)
y_test = np.argmax(y_test, axis=1)

# Accuracy
acc = accuracy_score(y_true=y_test, y_pred=y_pred)

# Get F1-scores for every class
f1_scores = f1_score(y_true=y_test, y_pred=y_pred, average=None)

# Get macro f1 score
macro_f1 = f1_score(y_true=y_test, y_pred=y_pred, average="macro")

print(f"Accuracy: {acc}")
print(f"F1-score for class 0: {f1_scores[0]}")
print(f"F1-score for class 1 (cough): {f1_scores[1]}")
print(f"F1-score for class 2: {f1_scores[2]}")
print(f"Macro F1-score: {macro_f1}")

print(classification_report(y_true=y_test, y_pred=y_pred))
print("Confusion matrix")
print(confusion_matrix(y_true=y_test, y_pred=y_pred))

sys.stdout.close()
