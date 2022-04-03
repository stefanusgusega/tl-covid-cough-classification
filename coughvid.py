"""
Main program for COUGHVID dataset
"""
from speechpy.processing import cmvn, cmvnw
from tqdm import tqdm
import argparse
import numpy as np
import os
import pickle as pkl
from src.train import Trainer

DUMP_PATH = "dumps"
LOG_PATH = os.path.join(DUMP_PATH, "logs")

with (open(os.path.join(DUMP_PATH, "features.pkl"), "rb")) as f:
    features = pkl.load(f)

print(features[0].shape)

cmvns = []
for feat in tqdm(features[0]):
    cmvned = cmvnw(feat, win_size=2047, variance_normalization=True)
    cmvns.append(cmvned)

cmvns = np.array(cmvns)

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
    audio_datas=cmvns,
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
