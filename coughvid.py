"""
Main program for COUGHVID dataset
"""
import argparse
import os
import sys
import pickle as pkl
from src.train import Trainer
from src.utils.chore import generate_now_datetime

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

DUMP_PATH = "dumps"
LOG_PATH = os.path.join(DUMP_PATH, "logs")
FEATURE_FILE = args.f

with (open(os.path.join(DUMP_PATH, FEATURE_FILE), "rb")) as f:
    features = pkl.load(f)

print("Dataset: COUGHVID")
print(f"Epoch: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Feature file: {FEATURE_FILE}")

model_args = {"input_shape": (features[0].shape[1], features[0].shape[2], 1)}

# TODO : HP Tuning doesn't work
hp_args = {
    # "first_dense_units" : [2**i for i in range(8,10)],
    "first_dense_units": [2**8],
    # "second_dense_units" : [2**i for i in range(3,5)],
    "second_dense_units": [2**3],
    "learning_rates": [1e-3],
    "batch_size": [128],
}

trainer = Trainer(
    audio_datas=features[0],
    audio_labels=features[1],
    model_args=model_args,
    tensorboard_log_dir=LOG_PATH,
    # hyperparameter_tuning_args=hp_args,
)

# trainer.set_tensorboard_callback(log_dir=LOG_PATH)
# trainer.train(
#     epochs=2, hp_model_tuning_folder=os.path.join(DUMP_PATH, "hyperparameter_models/")
# )

trainer.train(epochs=epochs, batch_size=batch_size)

sys.stdout.close()
