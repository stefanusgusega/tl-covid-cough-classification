"""
Other utility functions.
"""
from datetime import datetime
import os
import pickle as pkl


def generate_now_datetime():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def create_folder(parent_folder: str, prefix: str):
    folder_name = f"{prefix}_{generate_now_datetime()}"
    full_path = os.path.join(parent_folder, folder_name)
    os.mkdir(full_path)
    return full_path


def save_obj_to_pkl(to_save, file_path):
    with (open(file_path, "wb")) as f:
        pkl.dump(to_save, f)


def load_obj_from_pkl(file_path):
    with (open(file_path, "rb")) as f:
        return pkl.load(f)
