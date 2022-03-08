import os
from typing import Tuple
import numpy as np
import pandas as pd
import pickle as pkl
from utils.audio import (
    convert_audio_to_numpy,
    extract_melspec,
    generate_segmented_data,
    pad_audio_with_silence,
)


def preprocess_dataframe(
    df: pd.DataFrame,
    audio_folder_path: str,
    sampling_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = None,
    is_mel_spec: bool = True,
    save_to_pickle=True,
    backup_every_stage=True,
    pickle_folder=None,
) -> Tuple[np.ndarray, np.ndarray]:

    if save_to_pickle or backup_every_stage:
        # Check if the folder is specified, if not so raise an exception
        if pickle_folder is None:
            raise Exception(
                "Please specify the path that you wish to dump the results of this conversion."
            )

    numpy_data, covid_statuses = convert_audio_to_numpy(
        df, audio_folder_path=audio_folder_path, sampling_rate=sampling_rate
    )

    if backup_every_stage:
        print("Creating backup for numpy data...")
        save_obj_to_pkl(
            (numpy_data, covid_statuses), os.path.join(pickle_folder, "numpy_data.pkl")
        )
        print("Backup for numpy data created.")

    segmented_data, segmented_covid_status = generate_segmented_data(
        numpy_data, covid_statuses, sampling_rate=sampling_rate
    )

    if backup_every_stage:
        print("Creating backup for segmented data...")
        save_obj_to_pkl(
            (segmented_data, covid_statuses),
            os.path.join(pickle_folder, "segmented_data.pkl"),
        )
        print("Backup for segmented data created.")

    # Pad the data
    padded_data = pad_audio_with_silence(segmented_data)

    if backup_every_stage:
        print("Creating backup for padded data...")
        save_obj_to_pkl(
            (padded_data, covid_statuses),
            os.path.join(pickle_folder, "padded_data.pkl"),
        )
        print("Backup for padded data created.")

    # Data augmentation
    balanced_data = padded_data

    # Feature extraction
    if is_mel_spec:
        features = extract_melspec(
            balanced_data,
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

    # TODO: data augmenttion and MFCC

    res = features, segmented_covid_status

    # Save to pickle file
    if save_to_pickle:
        print("Saving features...")
        save_obj_to_pkl(res, os.path.join(pickle_folder, "features.pkl"))
        print("Features saved.")

    # Returning series of data in (-1, 1) shape and the labels in (-1, 1) too
    # NOW : features in 2D shape
    return res


def save_obj_to_pkl(to_save, file_path):
    with (open(file_path, "wb")) as f:
        pkl.dump(to_save, f)
