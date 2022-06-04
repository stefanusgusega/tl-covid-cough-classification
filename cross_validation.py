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
CHECKPOINT_PATH = os.path.join(DUMP_PATH, "checkpoints")

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

sys.stdout = open(
    os.path.join("logs", "coughvid", f"{generate_now_datetime()}.txt"),
    "w",
    encoding="utf-8",
)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int)
parser.add_argument("-d")
parser.add_argument("-bs", type=int)
args = parser.parse_args()

epochs = args.e
batch_size = args.bs

DATA_FILE = args.d

X_train, y_train = load_obj_from_pkl(
    os.path.join(DUMP_PATH, "data", "covid-train", DATA_FILE)
)

print(f"Epoch: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Data file: {DATA_FILE}")

# model_args = {"input_shape": (X_train.shape[1], X_train.shape[2], 1)}
log_dir = dict(tensorboard=LOG_PATH, checkpoint=CHECKPOINT_PATH)

trainer = Trainer(
    audio_datas=X_train,
    audio_labels=y_train,
    # model_args=model_args,
    log_dir=log_dir,
)
# trainer.set_early_stopping_callback()

trainer.cross_validation(
    epochs=epochs,
    batch_size=batch_size,
    feature_parameter=dict(n_mels=64, hop_length=128),
    other_model_args={},
)

sys.stdout.close()
