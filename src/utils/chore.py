"""
Other utility functions.
"""
from datetime import datetime
import pickle as pkl


def generate_now_datetime():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def save_obj_to_pkl(to_save, file_path):
    with (open(file_path, "wb")) as f:
        pkl.dump(to_save, f)


def load_obj_from_pkl(file_path):
    with (open(file_path, "rb")) as f:
        return pkl.load(f)
