"""
Data preprocessing util functions
"""
import os
from typing import Tuple
import pickle as pkl
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler
from src.utils.audio import (
    augment_data,
    convert_audio_to_numpy,
    equalize_audio_duration,
    extract_melspec,
    generate_segmented_data,
)


def preprocess_covid_dataframe(
    df: pd.DataFrame,
    audio_folder_path: str,
    sampling_rate: int = 16000,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = None,
    backup_every_stage=True,
    pickle_folder=None,
) -> Tuple[np.ndarray, np.ndarray]:

    if backup_every_stage:
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

    # Equalize the data duration
    equal_duration_data = equalize_audio_duration(segmented_data)

    if backup_every_stage:
        print("Creating backup for equal duration data...")
        save_obj_to_pkl(
            (equal_duration_data, segmented_covid_statuses),
            os.path.join(pickle_folder, "equal_duration_data.pkl"),
        )
        print("Backup for equal duration data created.")

    # ? This is the data augmentation
    # Append this augmented_data to actual data
    # balanced_data, balanced_covid_statuses = balance_data(
    #     equal_duration_data,
    #     segmented_covid_statuses,
    #     sampling_rate=sampling_rate,
    #     pickle_folder=pickle_folder,
    # )

    # Do undersampling
    # balanced_data, balanced_covid_statuses = undersample_data(
    #     audio_datas=equal_duration_data,
    #     labels=segmented_covid_statuses,
    #     pivot_label="COVID-19",
    # )
    balanced_data, balanced_covid_statuses = RandomUnderSampler(
        sampling_strategy="majority", random_state=42
    ).fit_resample(equal_duration_data, segmented_covid_statuses)

    # Backup the balancing data stage
    if backup_every_stage:
        print("Creating backup for balanced data...")
        save_obj_to_pkl(
            (balanced_data, balanced_covid_statuses),
            os.path.join(pickle_folder, "balanced_data.pkl"),
        )
        print("Backup for balanced data created.")

    # Feature extraction
    features = extract_melspec(
        balanced_data,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )

    # TODO: MFCC

    # NOW : returning features in 2D shape and status in 2D shape with one hot encoding fashion
    res = features, balanced_covid_statuses.reshape(-1, 1)

    # Save to pickle file for the final features
    print("Saving features...")
    save_obj_to_pkl(res, os.path.join(pickle_folder, "features.pkl"))
    print("Features saved.")

    # Should be returning series of data in (-1, 1) shape and the labels in (-1, 1) too
    return res


def balance_data(
    datas: np.ndarray, labels: np.ndarray, sampling_rate: int = 16000, **kwargs
):
    # Data augmentation
    # NOW : labels should be contains only 2 labels
    n_aug = get_pos_neg_diff(labels)

    # Data being augmented should contain only COVID-19 audio data
    covid_data_only = get_covid_data_only(datas, labels)

    # Augment the positive class
    augmented_data = augment_data(
        covid_data_only,
        n_aug=n_aug,
        sampling_rate=sampling_rate,
    )

    # IMPORTANT: For testing only, so augmented_data will be pickled
    save_obj_to_pkl(
        augmented_data, os.path.join(kwargs["pickle_folder"], "augmented_data.pkl")
    )
    augmented_covid_status = np.full(len(augmented_data), "COVID-19")

    # Append this augmented_data to actual data
    balanced_data, balanced_covid_statuses = np.concatenate(
        (datas, augmented_data)
    ), np.concatenate((labels, augmented_covid_status))

    return balanced_data, balanced_covid_statuses


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


def get_pos_neg_diff(status_datas: np.ndarray):
    # precondition : status datas should be binary
    _, unique_counts = np.unique(status_datas, return_counts=True)

    return abs(np.diff(unique_counts))[0]


def undersample_data(audio_datas: np.ndarray, labels: np.ndarray, pivot_label):
    n_data_pivot = np.count_nonzero(labels == pivot_label)

    new_datas = []
    new_labels = []
    for label in np.unique(labels):
        # Get the index of the label observed
        labels_idx = np.argwhere(labels == label).flatten()

        # Pick randomly where idx will be being kept when not pivot_label
        being_keep_idx = np.random.choice(labels_idx, size=n_data_pivot, replace=False)

        # And then take the correlated data
        # If not pivot label, based on random chosen data
        if label != pivot_label:
            new_data = audio_datas[being_keep_idx]
        # If pivot label, based on labels_idx
        else:
            new_data = audio_datas[labels_idx]

        # Append to the list
        new_datas.append(new_data)
        new_labels.append(np.full((n_data_pivot,), fill_value=label))

    return np.concatenate(np.array(new_datas)), np.array(new_labels).flatten()


def expand_mel_spec(old_mel_specs: np.ndarray):
    new_mel_specs = []

    for mel_spec in old_mel_specs:
        new_mel_spec = np.expand_dims(mel_spec, -1)
        new_mel_specs.append(new_mel_spec)

    return np.array(new_mel_specs)


def encode_label(labels: np.ndarray, pos_label: str = None):
    # If binary, encode it according to positive label and negative label first
    if len(np.unique(labels)) == 2:
        if pos_label is None:
            raise Exception("Please specify which is the positive label")
        labels = np.where(labels == pos_label, 1, 0).reshape(-1, 1)

    return to_categorical(labels)
