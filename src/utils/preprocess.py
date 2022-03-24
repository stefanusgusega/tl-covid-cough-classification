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

    numpy_data, covid_statuses = convert(
        df=df,
        audio_folder_path=audio_folder_path,
        sampling_rate=sampling_rate,
        pickle_folder=pickle_folder,
        backup_every_stage=backup_every_stage,
    )

    segmented_data, segmented_covid_statuses = segment(
        numpy_data=numpy_data,
        covid_statuses=covid_statuses,
        sampling_rate=sampling_rate,
        pickle_folder=pickle_folder,
        backup_every_stage=backup_every_stage,
    )

    equal_duration_data, segmented_covid_statuses = equalize(
        segmented_data=segmented_data,
        segmented_covid_statuses=segmented_covid_statuses,
        pickle_folder=pickle_folder,
        backup_every_stage=backup_every_stage,
    )

    balanced_data, balanced_covid_statuses = balance(
        equal_duration_data=equal_duration_data,
        segmented_covid_statuses=segmented_covid_statuses,
        pickle_folder=pickle_folder,
        backup_every_stage=backup_every_stage,
    )

    features = extract(
        balanced_data=balanced_data,
        sampling_rate=sampling_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        pickle_folder=pickle_folder,
    )

    # NOW : returning features in 2D shape and status in 2D shape with one hot encoding fashion
    res = features, balanced_covid_statuses.reshape(-1, 1)

    # Save to pickle file for the final features
    print("Saving features...")
    save_obj_to_pkl(res, os.path.join(pickle_folder, "features.pkl"))
    print("Features saved.")

    # Should be returning series of data in (-1, 1) shape and the labels in (-1, 1) too
    return res


def convert(df, audio_folder_path, sampling_rate, pickle_folder, backup_every_stage):
    try:
        print("Loading numpy data from 'numpy_data.pkl'...")
        numpy_data, covid_statuses = load_obj_from_pkl(
            os.path.join(pickle_folder, "numpy_data.pkl")
        )
        print("Numpy data loaded.")
    except FileNotFoundError:
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

    return numpy_data, covid_statuses


def segment(
    numpy_data, covid_statuses, sampling_rate, pickle_folder, backup_every_stage
):
    try:
        print("Loading segmented data from 'segmented_data.pkl'...")
        segmented_data, segmented_covid_statuses = load_obj_from_pkl(
            os.path.join(pickle_folder, "segmented_data.pkl")
        )
        print("Segmented data loaded.")
    except FileNotFoundError:
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

    return segmented_data, segmented_covid_statuses


def equalize(
    segmented_data, segmented_covid_statuses, pickle_folder, backup_every_stage
):
    # Equalize the data duration, the data should be produced from segmentation
    try:
        print("Loading equal duration data from 'equal_duration_data.pkl'...")
        equal_duration_data = load_obj_from_pkl(
            os.path.join(pickle_folder, "equal_duration_data.pkl")
        )[0]
        print("Equal duration data loaded.")
    except FileNotFoundError:
        equal_duration_data = equalize_audio_duration(segmented_data)

        if backup_every_stage:
            print("Creating backup for equal duration data...")
            save_obj_to_pkl(
                (equal_duration_data, segmented_covid_statuses),
                os.path.join(pickle_folder, "equal_duration_data.pkl"),
            )
            print("Backup for equal duration data created.")

    return equal_duration_data, segmented_covid_statuses


def balance(
    equal_duration_data, segmented_covid_statuses, pickle_folder, backup_every_stage
):
    try:
        print("Loading balanced data from 'balanced_data.pkl'...")
        balanced_data, balanced_covid_statuses = load_obj_from_pkl(
            os.path.join(pickle_folder, "balanced_data.pkl")
        )
        print("Balanced data loaded.")
    except FileNotFoundError:
        print("Balancing data using undersampling...")
        balanced_data, balanced_covid_statuses = RandomUnderSampler(
            sampling_strategy="majority", random_state=42
        ).fit_resample(equal_duration_data, segmented_covid_statuses)
        print("Data balanced.")

        # Backup the balancing data stage
        if backup_every_stage:
            print("Creating backup for balanced data...")
            save_obj_to_pkl(
                (balanced_data, balanced_covid_statuses),
                os.path.join(pickle_folder, "balanced_data.pkl"),
            )
            print("Backup for balanced data created.")
    return balanced_data, balanced_covid_statuses


def extract(balanced_data, sampling_rate, n_fft, hop_length, win_length, pickle_folder):
    # Feature extraction
    try:
        print("Loading features from 'features.pkl'...")
        features = load_obj_from_pkl(os.path.join(pickle_folder, "features.pkl"))[0]
        print("Features loaded.")
    except FileNotFoundError:
        features = extract_melspec(
            balanced_data,
            sampling_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

    # TODO: MFCC

    return features


def save_obj_to_pkl(to_save, file_path):
    with (open(file_path, "wb")) as f:
        pkl.dump(to_save, f)


def load_obj_from_pkl(file_path):
    with (open(file_path, "rb")) as f:
        return pkl.load(f)


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
