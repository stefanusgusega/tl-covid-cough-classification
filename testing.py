"""
Main program for testing model
"""
import argparse
import os
import sys
import tensorflow as tf

from src.utils.chore import generate_now_datetime, load_obj_from_pkl

DUMP_PATH = "dumps"
LOG_PATH = os.path.join(DUMP_PATH, "logs")
MODEL_DIR = os.path.join(DUMP_PATH, "models")
DATA_PATH = os.path.join(DUMP_PATH, "data")

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
X_test, y_test = load_obj_from_pkl(DATA_FILE)

# Preprocessing dataset


# Get all models
model_paths = [f for f in os.listdir(MODEL_DIR) if f.startswith(MODEL_PREFIX)]

# Iterate for all models to predict and count classification score
for model_path in model_paths:
    # Load model first
    model = tf.keras.models.load_model(os.path.join(MODEL_DIR, model_path))

    # Do prediction
    y_pred = model.predict(X_test)

    print(y_pred)

# Report the average of the accuracy, AUC, and F1 maybe?

sys.stdout.close()
