"""
Audio util functions.
"""

import os
from typing import Tuple
import numpy as np
import pandas as pd
import librosa
from audiomentations import TimeStretch, Gain, Compose
from scipy import signal
from tqdm import tqdm
from src.utils.chore import generate_now_datetime, save_obj_to_pkl

AUDIO_EXTENSIONS = ["wav", "mp3", "webm", "ogg"]


def generate_cough_segments(
    x,
    sampling_rate,
    # Coughvid : cough_padding=0.2
    cough_padding=0.1,
    min_cough_len=0.2,
    th_l_multiplier=0.1,
    th_h_multiplier=2,
):
    """
    Preprocess the data by segmenting each file into individual coughs
    using a hysteresis comparator on the signal power

    Inputs:
    * x (np.array): cough signal
    * sampling_rate (float): sampling frequency in Hz
    * cough_padding (float): number of seconds added to the beginning and end of each detected cough
    to make sure coughs are not cut short
    * min_cough_length (float): length of the minimum possible segment that can be considered a cough
    * th_l_multiplier (float): multiplier of the RMS energy used as a lower threshold of the hysteresis comparator
    * th_h_multiplier (float): multiplier of the RMS energy used as a high threshold of the hysteresis comparator

    Outputs:
    * coughSegments (np.array of np.arrays): a list of cough signal arrays corresponding to each cough
    * cough_mask (np.array): an array of booleans that are True at the indices where a cough is in progress

    Source : Coughvid Repository
    """

    cough_mask = np.array([False] * len(x))

    # Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    # seg_th_l = th_l_multiplier * rms
    # seg_th_h = th_h_multiplier * rms

    # Segment coughs
    cough_segments = []
    padding = round(sampling_rate * cough_padding)
    min_cough_samples = round(sampling_rate * min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    # Coughvid : round(0.01 * sr)
    tolerance = round(0.05 * sampling_rate)
    below_th_counter = 0

    for i, sample in enumerate(x**2):
        if cough_in_progress:
            if sample < (th_l_multiplier * rms):
                below_th_counter += 1
                if below_th_counter > tolerance:
                    cough_end = i + padding if (i + padding < len(x)) else len(x) - 1
                    cough_in_progress = False
                    if cough_end + 1 - cough_start - 2 * padding >= min_cough_samples:
                        cough_segments.append(x[cough_start : cough_end + 1])
                        cough_mask[cough_start : cough_end + 1] = True
            # Else if the end of samples
            elif i == (len(x) - 1):
                cough_end = i
                cough_in_progress = False
                if cough_end + 1 - cough_start - 2 * padding >= min_cough_samples:
                    cough_segments.append(x[cough_start : cough_end + 1])
            else:
                below_th_counter = 0
        else:
            if sample > (th_h_multiplier * rms):
                cough_start = i - padding if (i - padding >= 0) else 0
                cough_in_progress = True

    return cough_segments, cough_mask


def convert_audio_to_numpy(
    df: pd.DataFrame,
    audio_folder_path: str,
    checkpoint_folder_path: str,
    sampling_rate: int = 16000,
    df_args: dict = None,
    segment: bool = False,
    segment_args: dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If segment, please specify on ```segment_args``` the ```start_colname``` and ```end_colname```.
    """

    if df_args is None:
        df_args = dict(
            filename_colname="uuid", ext_colname="ext", label_colname="status"
        )

    samples = []
    statuses = []

    print("Converting audio to numpy array...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Check if the filename already including extension. If yes, then use it instead.
        if row[df_args["filename_colname"]].split(".")[-1] in AUDIO_EXTENSIONS:
            filename = row[df_args["filename_colname"]]
        else:
            filename = row[df_args["filename_colname"]] + row[df_args["ext_colname"]]

        # Sampling rate is not returned because it will make worse memory usage
        audio_data, _ = librosa.load(
            os.path.join(audio_folder_path, filename), sr=sampling_rate
        )

        # If directly segment, then should know the start sample and end sample as stated on df
        if segment:
            start_sample = round(sampling_rate * row[segment_args["start_colname"]])
            end_sample = round(sampling_rate * row[segment_args["end_colname"]])
            samples.append(audio_data[start_sample : end_sample + 1])
        else:
            samples.append(audio_data)
        statuses.append(row[df_args["label_colname"]])

        # Save the backup
        save_obj_to_pkl(
            (np.array(samples), np.array(statuses), idx),
            os.path.join(
                checkpoint_folder_path,
                f"numpy_data_{generate_now_datetime}/numpy_data.pkl",
            ),
        )

    return np.array(samples), np.array(statuses)


def segment_cough_and_label(
    original_audio: np.ndarray, covid_status: str, sampling_rate: int = 16000
) -> Tuple[np.ndarray, np.ndarray]:

    cough_segments, _ = generate_cough_segments(original_audio, sampling_rate)

    segments = [np.array(segment) for segment in cough_segments]

    return np.array(segments), np.full((len(cough_segments),), covid_status)


def generate_segmented_data(
    samples_data: np.ndarray,
    audio_labels: np.ndarray,
    sampling_rate: int = 16000,
) -> Tuple[np.ndarray, np.ndarray]:

    if len(samples_data) != len(audio_labels):
        raise Exception(
            f"The length of the samples data and the labels is not same. "
            f"{len(samples_data)} != {len(audio_labels)}"
        )

    new_data = []
    labels_data = []

    print("Segmenting audio data...")

    for data, status_data in tqdm(
        zip(samples_data, audio_labels), total=len(samples_data)
    ):
        segments, labels = segment_cough_and_label(
            data, status_data, sampling_rate=sampling_rate
        )

        for segment in segments:
            new_data.append(segment)

        # Append the segmented labels
        labels_data.append(labels)

    labels_data = np.array(labels_data)

    return np.array(new_data), np.concatenate(labels_data)


def equalize_audio_duration(audio_datas: np.ndarray) -> np.ndarray:
    # Get length of audio by mapping len function
    audio_length_func = np.vectorize(len)
    audio_length_arr = audio_length_func(audio_datas)

    # Set offset value by higher outlier
    first_quartile = np.percentile(audio_length_arr, 25)
    third_quartile = np.percentile(audio_length_arr, 75)
    iqr = third_quartile - first_quartile
    offset = round(third_quartile + 1.5 * iqr)

    new_audio_datas = []

    print("Equalizing data...")

    for audio_data in tqdm(audio_datas, total=len(audio_datas)):
        # If length less than offset, then center pad it
        if len(audio_data) <= offset:
            padded_data = librosa.util.pad_center(
                data=np.array(audio_data), size=offset
            )
            new_audio_datas.append(padded_data)
            continue

        # Else, trim it
        # Randomize the start state
        random_start = np.random.randint(0, len(audio_data) - offset)

        # Trim it
        new_audio_datas.append(audio_data[random_start : (random_start + offset)])

    print("Data equalized.")

    return np.array(new_audio_datas)


def augment_data(
    audio_datas: np.ndarray, n_aug: int, sampling_rate: int = 16000
) -> np.ndarray:
    # Precondition: all datas are from most discriminated data

    # Check n_aug is defined
    if n_aug is None:
        raise Exception(
            "Please specify n_aug or how many data you want to be augmented."
        )

    augmented_datas = []

    # Time stretch
    time_stretch_augment = Compose([TimeStretch(p=1)])

    # Gain
    gain_augment = Compose([Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1)])

    # Time stretch + gain
    ts_gain_augment = Compose(
        [TimeStretch(p=1), Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1)]
    )

    augment_array = [time_stretch_augment, gain_augment, ts_gain_augment]

    # TODO: should check the truthy of this code
    print("Augmenting data...")

    for _ in tqdm(range(n_aug)):
        random_audio_data = audio_datas[np.random.choice(len(audio_datas))]
        aug = np.random.choice(augment_array)(
            samples=random_audio_data, sample_rate=sampling_rate
        )
        augmented_datas.append(aug)

    return np.array(augmented_datas)


def extract_melspec(
    audio_datas: np.ndarray,
    sampling_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = None,
) -> np.ndarray:
    # Initiate new container
    mel_specs = []

    print("Extracting mel spectrograms from audio...")

    # For each audio
    for audio in tqdm(audio_datas):
        mel_spec = librosa.feature.melspectrogram(
            audio,
            sr=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=signal.windows.hamming,
        )

        # Change to decibels instead of power spectrogram
        log_mel_spec = librosa.power_to_db(mel_spec)

        mel_specs.append(log_mel_spec)

    return np.array(mel_specs)
