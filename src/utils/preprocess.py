"""
Data preprocessing util functions
"""
import os
from typing import Tuple
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler
from src.utils.audio import (
    convert_audio_from_df,
    convert_audio_from_folder,
    equalize_audio_duration,
    extract_melspec,
    generate_segmented_data,
)
from src.utils.chore import load_obj_from_pkl, save_obj_to_pkl

# Non-class Utils


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
        return labels

    return to_categorical(labels)


class Preprocessor:
    """
    This is the parent class of all preprocessor.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        filename_colname: str,
        label_colname: str,
        audio_folder_path: str,
        checkpoints: dict = None,
        backup_every_stage=True,
        pickle_folder=None,
        **kwargs
    ) -> None:
        self.df = df
        self.filename_colname = filename_colname
        self.label_colname = label_colname
        self.audio_folder_path = audio_folder_path
        self.backup_every_stage = backup_every_stage
        self.pickle_folder = pickle_folder

        self.current_data = None
        self.current_labels = None
        self.current_state = "initialized"

        if checkpoints is None or set(checkpoints.keys()) != set(
            [
                "numpy_data",
                "segment",
            ]
        ):
            checkpoints = dict(numpy_data=None, segment=None)

        self.checkpoints = checkpoints
        self.kwargs = kwargs

        if backup_every_stage:
            # Check if the folder is specified, if not so raise an exception
            if pickle_folder is None:
                raise Exception(
                    "Please specify the path that you wish to dump the results of this conversion."
                )

    def convert_to_numpy(self, sampling_rate: int = 16000):
        assert self.current_state == "initialized"

        try:
            print("Loading numpy data from 'numpy_data.pkl'...")
            numpy_data, labels = load_obj_from_pkl(
                os.path.join(self.pickle_folder, "numpy_data.pkl")
            )
            print("Numpy data loaded.")
        except FileNotFoundError:
            numpy_data, labels = convert_audio_from_df(
                self.df,
                audio_folder_path=self.audio_folder_path,
                checkpoint_folder_path=self.pickle_folder,
                sampling_rate=sampling_rate,
                df_args=dict(
                    filename_colname=self.filename_colname,
                    label_colname=self.label_colname,
                    ext_colname="ext",
                ),
                checkpoint=self.checkpoints["numpy_data"],
                **self.kwargs
            )

            if self.backup_every_stage:
                print("Creating backup for numpy data...")
                save_obj_to_pkl(
                    (numpy_data, labels),
                    os.path.join(self.pickle_folder, "numpy_data.pkl"),
                )
                print("Backup for numpy data created.")

        # Update state
        self.current_data = numpy_data
        self.current_labels = labels
        self.current_state = "converted"

        return self.current_data, self.current_labels

    def segment_audio(self, sampling_rate: int = 16000, sound_kind: str = "cough"):
        assert self.current_state == "converted"

        try:
            print("Loading segmented data from 'segmented_data.pkl'...")
            segmented_data, segmented_labels = load_obj_from_pkl(
                os.path.join(self.pickle_folder, "segmented_data.pkl")
            )
            print("Segmented data loaded.")
        except FileNotFoundError:
            segmented_data, segmented_labels = generate_segmented_data(
                self.current_data,
                self.current_labels,
                checkpoint_folder_path=self.pickle_folder,
                checkpoint=self.checkpoints["segment"],
                sampling_rate=sampling_rate,
                sound_kind=sound_kind,
            )

            if self.backup_every_stage:
                print("Creating backup for segmented data...")
                save_obj_to_pkl(
                    (segmented_data, segmented_labels),
                    os.path.join(self.pickle_folder, "segmented_data.pkl"),
                )
                print("Backup for segmented data created.")

        # Update data, labels, and state
        self.current_data = segmented_data
        self.current_labels = segmented_labels
        self.current_state = "segmented"

        return segmented_data, segmented_labels

    def equalize_duration(self):
        # Equalize the data duration, the data should be produced from segmentation
        # Should ensure that this is run from second_run() method
        # Because it changes the state to 'aggregated'
        assert self.current_state == "aggregated"

        try:
            print("Loading equal duration data from 'equal_duration_data.pkl'...")
            equal_duration_data = load_obj_from_pkl(
                os.path.join(self.pickle_folder, "equal_duration_data.pkl")
            )[0]
            print("Equal duration data loaded.")
        except FileNotFoundError:
            equal_duration_data = equalize_audio_duration(self.current_data)

            if self.backup_every_stage:
                print("Creating backup for equal duration data...")
                save_obj_to_pkl(
                    (equal_duration_data, self.current_labels),
                    os.path.join(self.pickle_folder, "equal_duration_data.pkl"),
                )
                print("Backup for equal duration data created.")

        # Update the data and state
        self.current_data = equal_duration_data
        self.current_state = "equalized"

        return equal_duration_data, self.current_labels

    def balance(self):
        assert self.current_state == "equalized"

        try:
            print("Loading balanced data from 'balanced_data.pkl'...")
            balanced_data, balanced_labels = load_obj_from_pkl(
                os.path.join(self.pickle_folder, "balanced_data.pkl")
            )
            print("Balanced data loaded.")
        except FileNotFoundError:
            print("Balancing data using undersampling...")
            balanced_data, balanced_labels = RandomUnderSampler(
                sampling_strategy="majority", random_state=42
            ).fit_resample(self.current_data, self.current_labels)
            print("Data balanced.")

            # Backup the balancing data stage
            if self.backup_every_stage:
                print("Creating backup for balanced data...")
                save_obj_to_pkl(
                    (balanced_data, balanced_labels),
                    os.path.join(self.pickle_folder, "balanced_data.pkl"),
                )
                print("Backup for balanced data created.")

        # Update the data, labels, and states
        self.current_data = balanced_data
        self.current_labels = balanced_labels
        self.current_state = "balanced"

        return balanced_data, balanced_labels

    def extract(self, **kwargs):
        # Feature extraction
        assert self.current_state == "balanced"

        try:
            print("Loading features from 'features.pkl'...")
            features = load_obj_from_pkl(
                os.path.join(self.pickle_folder, "features.pkl")
            )[0]
            print("Features loaded.")
        except FileNotFoundError:
            features = extract_melspec(self.current_data, **kwargs)

        res = features, self.current_labels.reshape(-1, 1)

        # Save to pickle file for the final features
        print("Saving features...")
        save_obj_to_pkl(res, os.path.join(self.pickle_folder, "features.pkl"))
        print("Features saved.")

        # Update data and state
        self.current_data, self.current_labels = res
        self.current_state = "extracted"

        return res

    def first_run(
        self, sampling_rate: int = 16000, sound_kind: str = "cough"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This is the process of preprocessing that should be
        run on single dataset, and then this method only produce the
        segmented audio data and segmented label.
        """
        # ! Just do until segment audio
        # ! and then produce the segmented audio
        self.convert_to_numpy(sampling_rate=sampling_rate)
        self.segment_audio(sampling_rate=sampling_rate, sound_kind=sound_kind)

        # Return series of data in (-1, 1) shape and the labels in (-1, 1) too
        return self.current_data, self.current_labels

    def second_run(self, aggregated_data, aggregated_labels, **kwargs):
        """
        This run should be run if the data and labels are agrregated
        from many datasets
        """
        self.current_data = aggregated_data
        self.current_labels = aggregated_labels
        self.current_state = "aggregated"

        self.equalize_duration()
        self.balance()
        self.extract(**kwargs)

        # Return series of data in (-1, 1) shape and the labels in (-1, 1) too
        return self.current_data, self.current_labels


class FolderDataPreprocessor(Preprocessor):
    """
    Preprocessor for data directly from folder
    """

    def convert_to_numpy(self, sampling_rate: int = 16000):
        assert self.current_state == "initialized"

        try:
            print("Loading numpy data from 'numpy_data.pkl'...")
            numpy_data, labels = load_obj_from_pkl(
                os.path.join(self.pickle_folder, "numpy_data.pkl")
            )
            print("Numpy data loaded.")
        except FileNotFoundError:
            numpy_data, labels = convert_audio_from_folder(
                self.audio_folder_path,
                sampling_rate=sampling_rate,
                checkpoint_folder_path=self.pickle_folder,
                checkpoint=self.checkpoints["numpy_data"],
            )

            if self.backup_every_stage:
                print("Creating backup for numpy data...")
                save_obj_to_pkl(
                    (numpy_data, labels),
                    os.path.join(self.pickle_folder, "numpy_data.pkl"),
                )
                print("Backup for numpy data created.")

        # Update state
        self.current_data = numpy_data
        self.current_labels = labels
        self.current_state = "converted"

        return self.current_data, self.current_labels
