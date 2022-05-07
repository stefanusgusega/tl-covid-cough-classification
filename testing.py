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
    f1_score,
    roc_auc_score,
)
import tensorflow as tf
from icecream import ic

from src.utils.preprocess import FeatureExtractor, encode_label, expand_mel_spec
from src.utils.chore import generate_now_datetime, load_obj_from_pkl

DUMP_PATH = "dumps"
LOG_PATH = os.path.join(DUMP_PATH, "logs")
MODEL_DIR = os.path.join(DUMP_PATH, "models")
DATA_PATH = os.path.join(DUMP_PATH, "data")
OFFSET = 13318

sys.stdout = open(
    os.path.join("logs", "testing", f"{generate_now_datetime()}.txt"),
    "w",
    encoding="utf-8",
)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-m")
parser.add_argument("-td")
args = parser.parse_args()

MODEL_PREFIX = args.m
DATA_FILE = os.path.join(DATA_PATH, args.td)

# Loading testing dataset
X_test_raw, y_test_raw = load_obj_from_pkl(DATA_FILE)

# Preprocessing dataset
extractor = FeatureExtractor(
    pickle_folder=os.path.join(DATA_PATH, "covid-test", "preprocessed"),
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
y_test = encode_label(y_test, pos_label="COVID-19")

# Expand the dimension of the data because ResNet expects 3D
X_test = expand_mel_spec(X_test)

# ic(X_test.shape)
# Get all models
model_paths = [f for f in os.listdir(MODEL_DIR) if f.startswith(MODEL_PREFIX)]
# ic(model_paths)
accs = []
f1_scores = []
aucs = []

# Iterate for all models to predict and count classification score
for model_path in model_paths:
    # Load model first
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, model_path))

    # Do prediction
    y_proba = model.predict(X_test)

    # Make label to 0 and 1
    y_pred = np.where(y_proba >= 0.5, 1, 0)

    # Accuracy
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    accs.append(acc)

    # ROC-AUC
    auc = roc_auc_score(y_true=y_test, y_score=y_proba.flatten())
    aucs.append(auc)

    # F1-score
    f1_score_ = f1_score(y_true=y_test, y_pred=y_pred)
    f1_scores.append(f1_score_)

    print(f"Accuracy: {acc}")
    print(f"F1-score: {f1_score_}")
    print(classification_report(y_true=y_test, y_pred=y_pred))


# Report the average of the accuracy, AUC, and F1 maybe?
print(f"Average accuracy: {np.mean(accs)}")
print(f"Average F1 score: {np.mean(f1_scores)}")
print(f"Average ROC-AUC score: {np.mean(aucs)}")

sys.stdout.close()
