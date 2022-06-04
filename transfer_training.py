"""
Main program for COVID classification
"""
import argparse
import os
import sys

from src.train import TransferLearningTrainer
from src.utils.chore import generate_now_datetime, load_obj_from_pkl

DUMP_PATH = "dumps"
LOG_PATH = os.path.join(DUMP_PATH, "logs")
CHECKPOINT_PATH = os.path.join(DUMP_PATH, "checkpoints")
PRETRAINED_MODEL_PATH = os.path.join(DUMP_PATH, "models", "pretrained_model")
MODEL_DIR = os.path.join(DUMP_PATH, "models")
PREFIX = "best_exp_feat"

sys.stdout = open(
    os.path.join("logs", "coughvid", f"{PREFIX}_{generate_now_datetime()}.txt"),
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
    # os.path.join(DUMP_PATH, "data", "pretrain-train", DATA_FILE)
)

print(f"Epoch: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Data file: {DATA_FILE}")

# model_args = {"input_shape": (X_train.shape[1], X_train.shape[2], 1)}
log_dir = dict(tensorboard=LOG_PATH, checkpoint=CHECKPOINT_PATH)

trainer = TransferLearningTrainer(
    audio_datas=X_train,
    audio_labels=y_train,
    # model_args=model_args,
    log_dir=log_dir,
)

other_model_args = dict(
    mode="feat_ext",
    pretrained_model_path=PRETRAINED_MODEL_PATH,
    open_layer=1,
)
# trainer.set_early_stopping_callback()

trainer.train(
    epochs=epochs,
    batch_size=batch_size,
    feature_parameter=dict(n_mels=64, hop_length=128),
    model_filepath=os.path.join(MODEL_DIR, PREFIX),
    other_model_args=other_model_args,
)

sys.stdout.close()
