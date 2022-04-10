"""
Main program for COUGHVID dataset
"""
import argparse
import os
import sys

from src.train import Trainer
from src.utils.chore import generate_now_datetime, load_obj_from_pkl

DUMP_PATH = "dumps"
LOG_PATH = os.path.join(DUMP_PATH, "logs")


sys.stdout = open(
    os.path.join("logs", "coughvid", f"{generate_now_datetime()}.txt"),
    "w",
    encoding="utf-8",
)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int)
parser.add_argument("-f")
parser.add_argument("-bs", type=int)
args = parser.parse_args()

epochs = args.e
batch_size = args.bs

FEATURE_FILE = args.f

X_train, y_train = load_obj_from_pkl(
    os.path.join(DUMP_PATH, "data", "covid-train", FEATURE_FILE)
)

print(f"Epoch: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Feature file: {FEATURE_FILE}")

model_args = {"input_shape": (X_train.shape[1], X_train.shape[2], 1)}

trainer = Trainer(
    audio_datas=X_train,
    audio_labels=y_train,
    model_args=model_args,
    tensorboard_log_dir=LOG_PATH,
)
# trainer.set_early_stopping_callback()
trainer.set_checkpoint_callback(DUMP_PATH)

trainer.train(epochs=epochs, batch_size=batch_size)

sys.stdout.close()
