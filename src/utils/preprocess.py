"""
Data preprocessing util functions
"""
from abc import abstractmethod
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
    generate_augmented_data,
    generate_segmented_data,
)
from src.utils.chore import diff, load_obj_from_pkl, save_obj_to_pkl

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

    def __init__(self, backup_every_stage=True, pickle_folder=None) -> None:
        self.backup_every_stage = backup_every_stage
        self.pickle_folder = pickle_folder

        self.current_data = None
        self.current_labels = None
        self.current_state = "initialized"

        if backup_every_stage:
            # Check if the folder is specified, if not so raise an exception
            if pickle_folder is None:
                raise Exception(
                    "Please specify the path that you wish to dump the results of this conversion."
                )


class DataSegmenter(Preprocessor):
    """
    Audio loader and segmentation class
    """

    def __init__(
        self,
        audio_folder_path: str,
        checkpoints: dict = None,
        backup_every_stage=True,
        pickle_folder=None,
    ) -> None:
        super().__init__(
            backup_every_stage=backup_every_stage,
            pickle_folder=pickle_folder,
        )
        if checkpoints is None or set(checkpoints.keys()) != set(
            [
                "numpy_data",
                "segment",
            ]
        ):
            checkpoints = dict(numpy_data=None, segment=None)

        self.audio_folder_path = audio_folder_path
        self.checkpoints = checkpoints

    @abstractmethod
    def convert_to_numpy(self, sampling_rate: int = 16000, **kwargs):
        assert self.current_state == "initialized"

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

    def run(
        self, sampling_rate: int = 16000, sound_kind: str = "cough", **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This is the process of preprocessing that should be
        run on single dataset, and then this method only produce the
        segmented audio data and segmented label.
        """
        # ! Just do until segment audio
        # ! and then produce the segmented audio
        self.convert_to_numpy(sampling_rate=sampling_rate, **kwargs)
        self.segment_audio(sampling_rate=sampling_rate, sound_kind=sound_kind)

        # Return series of data in (-1, 1) shape and the labels in (-1, 1) too
        return self.current_data, self.current_labels


class DataframeBasedSegmenter(DataSegmenter):
    """
    Preprocessor for data that referenced by a dataframe.
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
    ) -> None:
        super().__init__(
            audio_folder_path=audio_folder_path,
            checkpoints=checkpoints,
            backup_every_stage=backup_every_stage,
            pickle_folder=pickle_folder,
        )

        self.df = df
        self.filename_colname = filename_colname
        self.label_colname = label_colname

        self.current_data = None

    def convert_to_numpy(self, sampling_rate: int = 16000, **kwargs):
        super().convert_to_numpy(sampling_rate=sampling_rate, **kwargs)

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
                **kwargs
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


class FolderDataSegmenter(DataSegmenter):
    """
    Preprocessor for data directly from folder
    """

    def __init__(
        self,
        audio_folder_path: str,
        checkpoints: dict = None,
        backup_every_stage=True,
        pickle_folder=None,
    ) -> None:
        super().__init__(
            audio_folder_path=audio_folder_path,
            checkpoints=checkpoints,
            backup_every_stage=backup_every_stage,
            pickle_folder=pickle_folder,
        )

    def convert_to_numpy(self, sampling_rate: int = 16000, **kwargs):
        super().convert_to_numpy(sampling_rate, **kwargs)

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


class FeatureExtractor(Preprocessor):
    """
    The class is for balancing data and extracting features.
    """

    def __init__(
        self,
        backup_every_stage=True,
        pickle_folder=None,
        offset: int = None,
        for_training=True,
        oversample=False,
    ) -> None:
        super().__init__(
            backup_every_stage=backup_every_stage, pickle_folder=pickle_folder
        )

        self.offset = offset
        self.for_training = for_training
        self.oversample = oversample

    def equalize_duration(self):
        assert self.current_state == "aggregated"

        offset, equal_duration_data = equalize_audio_duration(
            self.current_data, offset=self.offset
        )

        # Save offset value for this dataset
        self.offset = offset

        # Update the data and state
        self.current_data = equal_duration_data
        self.current_state = "equalized"

        return self.current_data, self.current_labels

    def balance(self):
        assert self.current_state == "equalized"

        if self.oversample:
            print("Balancing data using data augmentation (oversampling)...")
            # TODO : check oversampling
            # Get which labels need to be oversampled
            labels, labels_count = np.unique(self.current_labels, return_counts=True)

            most_data = np.max(labels_count)
            new_data = []
            new_labels = []

            # For each label get the number of data needed to be balanced
            for label, label_count in zip(labels, labels_count):
                diff_with_most = diff(label_count, most_data)

                # If no difference with the most of data, then no need for augmentation
                if diff_with_most == 0:
                    continue

                # Get the data of the specified class ONLY
                indices = np.where(self.current_labels == label)

                # Generate augment
                augmented_data = generate_augmented_data(
                    audio_datas=self.current_data[indices], n_aug=diff_with_most
                )

                # Save augmented data
                new_data.append(augmented_data)
                new_labels.append(np.full(shape=(diff_with_most), fill_value=label))

            # Concat with current data and label
            balanced_data = np.concatenate(
                (self.current_data, np.concatenate(new_data))
            )
            balanced_labels = np.concatenate(
                (self.current_labels, np.concatenate(new_labels))
            )

        else:
            print("Balancing data using undersampling...")
            balanced_data, balanced_labels = RandomUnderSampler(
                sampling_strategy="majority", random_state=42
            ).fit_resample(self.current_data, self.current_labels)
        print("Data balanced.")

        # Update the data, labels, and states
        self.current_data = balanced_data
        self.current_labels = balanced_labels
        self.current_state = "balanced"

        return self.current_data, self.current_labels

    def extract(self, **kwargs):
        # Feature extraction
        # If current state is balanced, so this is intended for training
        # If equalized, this is intended for testing
        assert self.current_state in ["balanced", "equalized"]

        features = extract_melspec(self.current_data, **kwargs)

        res = features, self.current_labels.reshape(-1, 1)

        # Update data and state
        self.current_data, self.current_labels = res
        self.current_state = "extracted"

        return res

    def run(self, aggregated_data, aggregated_labels, **kwargs):
        """
        This run should be run if the data and labels are agrregated
        from many datasets
        """
        self.current_data = aggregated_data
        self.current_labels = aggregated_labels
        self.current_state = "aggregated"

        self.equalize_duration()

        if self.for_training:
            self.balance()

        self.extract(**kwargs)

        # Return series of data in (-1, 1) shape and the labels in (-1, 1) too
        return self.current_data, self.current_labels
