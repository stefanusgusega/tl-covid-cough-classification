from typing import Tuple
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
import os
import pickle as pkl


def generate_cough_segments(
    x, fs, cough_padding=0.2, min_cough_len=0.2, th_l_multiplier=0.1, th_h_multiplier=2
):
    """Preprocess the data by segmenting each file into individual coughs using a hysteresis comparator on the signal power

    Inputs:
    *x (np.array): cough signal
    *fs (float): sampling frequency in Hz
    *cough_padding (float): number of seconds added to the beginning and end of each detected cough to make sure coughs are not cut short
    *min_cough_length (float): length of the minimum possible segment that can be considered a cough
    *th_l_multiplier (float): multiplier of the RMS energy used as a lower threshold of the hysteresis comparator
    *th_h_multiplier (float): multiplier of the RMS energy used as a high threshold of the hysteresis comparator

    Outputs:
    *coughSegments (np.array of np.arrays): a list of cough signal arrays corresponding to each cough
    cough_mask (np.array): an array of booleans that are True at the indices where a cough is in progress

    Source : Coughvid Repository
    """

    cough_mask = np.array([False] * len(x))

    # Define hysteresis thresholds
    rms = np.sqrt(np.mean(np.square(x)))
    seg_th_l = th_l_multiplier * rms
    seg_th_h = th_h_multiplier * rms

    # Segment coughs
    cough_segments = []
    padding = round(fs * cough_padding)
    min_cough_samples = round(fs * min_cough_len)
    cough_start = 0
    cough_end = 0
    cough_in_progress = False
    tolerance = round(0.01 * fs)
    below_th_counter = 0

    for i, sample in enumerate(x**2):
        if cough_in_progress:
            if sample < seg_th_l:
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
            if sample > seg_th_h:
                cough_start = i - padding if (i - padding >= 0) else 0
                cough_in_progress = True

    return cough_segments, cough_mask


def convert_audio_to_numpy(
    df: pd.DataFrame,
    audio_folder_path: str,
    sampling_rate: int = 16000,
    filename_colname: str = "uuid",
    ext_colname: str = "ext",
    covid_status_colname: str = "status",
) -> Tuple[np.ndarray, np.ndarray]:

    samples = []
    statuses = []

    print("Converting audio to numpy array...")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row[filename_colname] + row[ext_colname]
        # Sampling rate is not returned because it will make worse memory usage
        audio_data, _ = librosa.load(
            os.path.join(audio_folder_path, filename), sr=sampling_rate
        )
        samples.append(audio_data)
        statuses.append(row[covid_status_colname])

    return np.array(samples), np.array(statuses)


def segment_cough_and_label(
    original_audio: np.ndarray, covid_status: str, sampling_rate: int = 16000
) -> Tuple[np.ndarray, np.ndarray]:

    cough_segments, _ = generate_cough_segments(original_audio, sampling_rate)

    segments = [np.array(segment) for segment in cough_segments]

    return np.array(segments), np.full((len(cough_segments),), covid_status)


def generate_segmented_data(
    samples_data: np.ndarray, covid_status_data: np.ndarray, sampling_rate: int = 16000
) -> Tuple[np.ndarray, np.ndarray]:

    if len(samples_data) != len(covid_status_data):
        raise Exception(
            f"The length of the samples data and covid status data is not same. {len(samples_data)} != {len(covid_status_data)}"
        )

    new_data = []
    statuses_data = []

    print("Segmenting audio data...")

    for data, status_data in tqdm(
        zip(samples_data, covid_status_data), total=len(samples_data)
    ):
        segments, labels = segment_cough_and_label(
            data, status_data, sampling_rate=sampling_rate
        )
        new_data.append(segments)
        status_data.append(labels)

    new_data = np.array(new_data)
    statuses_data = np.array(statuses_data)

    return np.concatenate(new_data).reshape(-1, 1), np.concatenate(status_data).reshape(
        -1, 1
    )
