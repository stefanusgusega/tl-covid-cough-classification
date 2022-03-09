import os
from typing import Tuple
import numpy as np
import pandas as pd
import pickle as pkl
from utils.audio import (
    augment_data,
    convert_audio_to_numpy,
    extract_melspec,
    generate_segmented_data,
    pad_audio_with_silence,
)
from utils.dataframe import get_pos_neg_diff


def preprocess_covid_dataframe(
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
            (numpy_data, covid_statuses),
            os.path.join(pickle_folder, "numpy_data.pkl"),
        )
        print("Backup for numpy data created.")

    segmented_data, segmented_covid_statuses = generate_segmented_data(
        numpy_data, covid_statuses, sampling_rate=sampling_rate
    )

    if backup_every_stage:
        print("Creating backup for segmented data...")
        save_obj_to_pkl(
            (segmented_data, segmented_covid_statuses),
            os.path.join(pickle_folder, "segmented_data.pkl"),
        )
        print("Backup for segmented data created.")

    # Pad the data
    padded_data = pad_audio_with_silence(segmented_data)

    if backup_every_stage:
        print("Creating backup for padded data...")
        save_obj_to_pkl(
            (padded_data, segmented_covid_statuses),
            os.path.join(pickle_folder, "padded_data.pkl"),
        )
        print("Backup for padded data created.")

    # Data augmentation
    n_aug = get_pos_neg_diff(df, "status")

    # Data being augmented should contain only COVID-19 audio data
    covid_data_only = get_covid_data_only(padded_data, segmented_covid_statuses)

    # Augment the positive class
    augmented_data = augment_data(
        covid_data_only, n_aug=n_aug, sampling_rate=sampling_rate
    )

    # IMPORTANT: For testing only, so augmented_data will be pickled
    save_obj_to_pkl(augmented_data, os.path.join(pickle_folder, "augmented_data.pkl"))
    augmented_covid_status = np.full(len(augmented_data), "COVID-19")

    # Append this augmented_data to actual data
    balanced_data, balanced_covid_statuses = np.concatenate(
        (covid_data_only, augmented_data)
    ), np.concatenate((segmented_covid_statuses, augmented_covid_status))

    # Backup the data augmentation stage
    if backup_every_stage:
        print("Creating backup for balanced data...")
        save_obj_to_pkl(
            (balanced_data, balanced_covid_statuses),
            os.path.join(pickle_folder, "balanced_data.pkl"),
        )
        print("Backup for balanced data created.")

    # Feature extraction
    if is_mel_spec:
        features = extract_melspec(
            balanced_data,
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

    # TODO: MFCC

    # NOW : returning features in 2D shape and status in 1D shape
    res = features, balanced_covid_statuses.reshape(-1, 1)

    # Save to pickle file for the final features
    if save_to_pickle:
        print("Saving features...")
        save_obj_to_pkl(res, os.path.join(pickle_folder, "features.pkl"))
        print("Features saved.")

    # Should be returning series of data in (-1, 1) shape and the labels in (-1, 1) too
    return res


def save_obj_to_pkl(to_save, file_path):
    with (open(file_path, "wb")) as f:
        pkl.dump(to_save, f)


def get_covid_data_only(
    audio_datas: np.ndarray, status_datas: np.ndarray
) -> np.ndarray:
    covid_only_data = []
    for audio, status in zip(audio_datas, status_datas):
        if status == "COVID-19":
            covid_only_data.append(audio)

    return np.array(covid_only_data)
