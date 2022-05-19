"""
Main program for pretraining
"""
import argparse
import os
import sys

from src.train import Pretrainer
from src.utils.chore import generate_now_datetime, load_obj_from_pkl

DUMP_PATH = "dumps"
LOG_PATH = os.path.join(DUMP_PATH, "logs")
CHECKPOINT_PATH = os.path.join(DUMP_PATH, "checkpoints")
MODEL_DIR = os.path.join(DUMP_PATH, "models")
FEATURE_EXTRACTOR_DIR = os.path.join(DUMP_PATH, "feature-extractors")

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

sys.stdout = open(
    os.path.join("logs", "pretraining", f"{generate_now_datetime()}.txt"),
    "w",
    encoding="utf-8",
)

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int)
parser.add_argument("-bs", type=int)
args = parser.parse_args()

epochs = args.e
batch_size = args.bs


X_train, y_train = load_obj_from_pkl(
    os.path.join(DUMP_PATH, "data", "pretrain-train", "pretrain_train.pkl")
)

print(f"Epoch: {epochs}")
print(f"Batch size: {batch_size}")

# model_args = {"input_shape": (X_train.shape[1], X_train.shape[2], 1)}
log_dir = dict(tensorboard=LOG_PATH, checkpoint=CHECKPOINT_PATH)

trainer = Pretrainer(
    audio_datas=X_train,
    audio_labels=y_train,
    # model_args=model_args,
    log_dir=log_dir,
)
# trainer.set_early_stopping_callback()

feature_extractor, model = trainer.train(
    epochs=epochs,
    batch_size=batch_size,
    feature_parameter=dict(n_mels=64, hop_length=128),
    model_filepath=os.path.join(MODEL_DIR, "pretrained_model"),
    feature_extractor_filepath=os.path.join(
        FEATURE_EXTRACTOR_DIR, "pretrained_feature_extractor.pkl"
    ),
)

sys.stdout.close()
