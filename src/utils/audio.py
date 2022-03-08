from audiomentations import TimeStretch, Gain, Compose
from typing import Tuple
from scipy import signal
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
import os
import random


def generate_cough_segments(
    x, fs, cough_padding=0.1, min_cough_len=0.2, th_l_multiplier=0.1, th_h_multiplier=2
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
    tolerance = round(0.05 * fs)
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
        statuses_data.append(labels)

    new_data = np.array(new_data)
    statuses_data = np.array(statuses_data)

    return np.concatenate(new_data), np.concatenate(statuses_data)


def pad_audio_with_silence(audio_datas: np.ndarray) -> np.ndarray:
    # Search for the longest duration
    # Map to audio length
    audio_length = lambda x: len(x)
    audio_length_func = np.vectorize(audio_length)
    audio_length_arr = audio_length_func(audio_datas)

    # Get the max length
    max_length = max(audio_length_arr)

    new_audio_datas = []

    print("Padding audio with silence...")

    # For each audio
    for audio in tqdm(audio_datas):
        # Pad the beginning audio
        pad_begin_len = random.randint(0, max_length - len(audio))

        # Pad the ending audio
        pad_end_len = max_length - len(audio) - pad_begin_len

        # Create zeros array
        pad_begin = np.zeros(pad_begin_len)
        pad_end = np.zeros(pad_end_len)

        # Concat
        new_data = np.concatenate((pad_begin, audio, pad_end))

        # Append
        new_audio_datas.append(new_data)

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

        # Change to decibels instead of amplitude
        log_mel_spec = librosa.power_to_db(mel_spec)

        mel_specs.append(log_mel_spec)

    return np.array(mel_specs)
