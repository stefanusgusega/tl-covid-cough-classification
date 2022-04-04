"""
Main program for COUGHVID dataset
"""
import argparse
import os
import sys
import pickle as pkl
import numpy as np
from src.train import Trainer
from src.utils.chore import generate_now_datetime

sys.stdout = open(
    os.path.join("logs", "coughvid", f"{generate_now_datetime()}.txt"),
    "w",
    encoding="utf-8",
)
DUMP_PATH = "dumps"
LOG_PATH = os.path.join(DUMP_PATH, "logs")

with (open(os.path.join(DUMP_PATH, "features.pkl"), "rb")) as f:
    features = pkl.load(f)

print(features[0].shape)

print((np.max(features[0]), np.min(features[0])))

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
parser = argparse.ArgumentParser()
parser.add_argument("-e", type=int)
parser.add_argument("-bs", type=int)
args = parser.parse_args()
trainer.train(epochs=args.e, batch_size=args.bs)

sys.stdout.close()
