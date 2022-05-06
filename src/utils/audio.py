"""
Audio util functions.
"""

import os
from typing import Tuple
from audioread.exceptions import NoBackendError
from speechpy.processing import cmvnw
import numpy as np
import pandas as pd
import librosa
from audiomentations import AddGaussianNoise, Gain, Compose

# from icecream import ic
from scipy import signal
from tqdm import tqdm
from src.utils.chore import create_folder, diff, save_obj_to_pkl

AUDIO_EXTENSIONS = ["wav", "mp3", "webm", "ogg", "flac", "m4a", "aiff"]


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


def generate_sneeze_segments(
    x,
    sampling_rate: int = 16000,
    padding: float = 0.1,
    min_sneeze_len: float = 0.3,
):
    # First, split the audio based on the top db
    non_silent_indices = librosa.effects.split(x, top_db=20)

    # Check if the silence duration of each split is below threshold
    # Threshold set as minimum sneeze len
    new_non_silent_indices = []

    # Should check if the indices produced more than one indice tuple
    if len(non_silent_indices) > 1:
        for idx, (i, j) in enumerate(
            zip(non_silent_indices[:-1], non_silent_indices[1:])
        ):
            # If the difference of end and start between two consecutive segments is below the threshold
            # Then merge those segments
            if diff(i[1], j[0]) <= min_sneeze_len * sampling_rate:
                new_non_silent_indices.append([i[0], j[1]])

            # If not, then just append the element from first array
            else:
                new_non_silent_indices.append([i[0], i[1]])
                # If last iteration, then append the element from second array too
                # So, subtract by 2, because zipping and index starts from 0
                if idx == len(non_silent_indices) - 2:
                    new_non_silent_indices.append([j[0], j[1]])

    # If not more than one indice tuple, then use the same non silent indices
    else:
        new_non_silent_indices = non_silent_indices

    # Convert padding to number of samples
    padding_samples = round(padding * sampling_rate)

    # Init sneeze segments array
    sneeze_segments = []

    # ic(new_non_silent_indices)

    # Iterate the indices
    for indice_tuple in np.array(new_non_silent_indices):
        # Check if the indice reached the minimum len
        if diff(indice_tuple[0], indice_tuple[1]) >= min_sneeze_len * sampling_rate:
            # Pad the start
            start = indice_tuple[0] - padding_samples
            # Pad the end
            end = indice_tuple[1] + padding_samples
            sneeze_segments.append(x[max(0, start) : min(len(x), end)])

    return np.array(sneeze_segments)


def convert_audio_from_df(
    df: pd.DataFrame,
    audio_folder_path: str,
    checkpoint_folder_path: str,
    checkpoint: dict = None,
    sampling_rate: int = 16000,
    df_args: dict = None,
    segment: bool = False,
    segment_args: dict = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If segment, please specify on ```segment_args``` the ```start_colname``` and ```end_colname```.
    """

    # If df_args is't specified or the keys are incomplete,
    # then use this defaults
    if df_args is None or set(df_args.keys()) != set(
        ["filename_colname", "ext_colname", "label_colname"]
    ):
        df_args = dict(
            filename_colname="uuid",
            ext_colname="ext",
            label_colname="status",
        )

    # Check if checkpoint is isn't specified or the keys are incomplete,
    # Apply defaults
    if checkpoint is None or set(checkpoint.keys()) != set(
        ["datas", "labels", "last_index"]
    ):
        checkpoint = dict(datas=[], labels=[], last_index=-1)

    df = df.loc[checkpoint["last_index"] + 1 :]
    samples = list(checkpoint["datas"])
    statuses = list(checkpoint["labels"])

    # Create new folder to save checkpoint
    specific_ckpt_folder_name = create_folder(checkpoint_folder_path, "numpy_data_ckpt")

    print("Converting audio to numpy array...")

    # Start tqdm from initial checkpoint state, or if there's no ckpt, it will start from 0
    for idx, row in tqdm(
        df.iterrows(), initial=len(statuses), total=(len(statuses) + len(df))
    ):
        # Check if the filename already including extension. If yes, then use it instead.
        if row[df_args["filename_colname"]].split(".")[-1] in AUDIO_EXTENSIONS:
            filename = row[df_args["filename_colname"]]
        else:
            filename = row[df_args["filename_colname"]] + row[df_args["ext_colname"]]

        try:
            # Sampling rate is not returned because it will make worse memory usage
            audio_data, _ = librosa.load(
                os.path.join(audio_folder_path, filename), sr=sampling_rate
            )

            # If directly segment, then should know the start sample and end sample as stated on df
            if segment:
                sample_range = dict(
                    start=round(sampling_rate * row[segment_args["start_colname"]]),
                    end=round(sampling_rate * row[segment_args["end_colname"]]),
                )

                samples.append(
                    audio_data[sample_range["start"] : sample_range["end"] + 1]
                )
            else:
                samples.append(audio_data)
            statuses.append(row[df_args["label_colname"]])

            # Save the backup to the created specific checkpoint folder
            save_obj_to_pkl(
                dict(
                    datas=np.array(samples), labels=np.array(statuses), last_index=idx
                ),
                os.path.join(
                    checkpoint_folder_path,
                    f"{specific_ckpt_folder_name}/numpy_data.pkl",
                ),
            )
        except (ValueError, FileNotFoundError, RuntimeError, NoBackendError) as e:
            print(f"Error occured on idx {idx}: {str(e)}")

    return np.array(samples), np.array(statuses)


def convert_audio_from_folder(
    audio_folder_path: str,
    sampling_rate: int = 16000,
    name: str = None,
    checkpoint: dict = None,
    checkpoint_folder_path: str = None,
):
    # If name is not specified, then set it with the name of the folder
    if name is None:
        name = os.path.basename(audio_folder_path)

    # Check if checkpoint is isn't specified or the keys are incomplete,
    # Apply defaults
    if checkpoint is None or set(checkpoint.keys()) != set(["datas", "last_index"]):
        checkpoint = dict(datas=[], last_index=-1)

    samples = list(checkpoint["datas"])

    # Create new folder to save checkpoint
    specific_ckpt_folder_name = create_folder(checkpoint_folder_path, "numpy_data_ckpt")
    samples = []

    for idx, audio_file in tqdm(
        enumerate(os.listdir(audio_folder_path)[checkpoint["last_index"] + 1 :])
    ):
        # If not an audio file, skip to next file
        if audio_file.split(".")[-1] not in AUDIO_EXTENSIONS:
            continue

        try:
            # Sampling rate is not returned because it will make worse memory usage
            audio_data, _ = librosa.load(
                os.path.join(audio_folder_path, audio_file), sr=sampling_rate
            )

            samples.append(audio_data)

            # Save the backup to the created specific checkpoint folder
            save_obj_to_pkl(
                dict(
                    datas=np.array(samples),
                    last_index=idx,
                ),
                os.path.join(
                    checkpoint_folder_path,
                    f"{specific_ckpt_folder_name}/numpy_data.pkl",
                ),
            )
        except (ValueError, FileNotFoundError, RuntimeError, NoBackendError) as e:
            print(f"Error occured on idx {idx}: {str(e)}")

    return np.array(samples), np.full((len(samples),), name)


def segment_cough_and_label(
    original_audio: np.ndarray,
    covid_status: str,
    sampling_rate: int = 16000,
    sound_kind: str = "cough",
) -> Tuple[np.ndarray, np.ndarray]:
    if sound_kind == "cough":
        cough_segments, _ = generate_cough_segments(original_audio, sampling_rate)
    elif sound_kind == "sneeze":
        cough_segments = generate_sneeze_segments(original_audio, sampling_rate)
    segments = [np.array(segment) for segment in cough_segments]

    return np.array(segments), np.full((len(cough_segments),), covid_status)


def generate_segmented_data(
    samples_data: np.ndarray,
    audio_labels: np.ndarray,
    checkpoint_folder_path: str,
    checkpoint: dict = None,
    sampling_rate: int = 16000,
    sound_kind: str = "cough",
) -> Tuple[np.ndarray, np.ndarray]:

    if len(samples_data) != len(audio_labels):
        raise Exception(
            f"The length of the samples data and the labels is not same. "
            f"{len(samples_data)} != {len(audio_labels)}"
        )

    # Check if checkpoint is isn't specified or the keys are incomplete,
    # Apply defaults
    if checkpoint is None or set(checkpoint.keys()) != set(
        ["datas", "labels", "last_index"]
    ):
        checkpoint = dict(datas=[], labels=[], last_index=-1)

    samples_data = samples_data[checkpoint["last_index"] + 1 :]
    new_data = list(checkpoint["datas"])
    labels_data = list(checkpoint["labels"])

    # Create new folder to save checkpoint
    specific_ckpt_folder_name = create_folder(checkpoint_folder_path, "segmented_ckpt")

    print("Segmenting audio data...")

    for idx, (data, status_data) in tqdm(
        enumerate(zip(samples_data, audio_labels)),
        initial=len(labels_data),
        total=(len(labels_data) + len(samples_data)),
    ):
        segments, labels = segment_cough_and_label(
            data, status_data, sampling_rate=sampling_rate, sound_kind=sound_kind
        )

        for segment in segments:
            new_data.append(segment)

        # Append the segmented labels
        labels_data.append(labels)

        # Checkpointing
        save_obj_to_pkl(
            dict(
                datas=np.array(new_data),
                labels=np.concatenate(np.array(labels_data)),
                last_index=idx,
            ),
            os.path.join(
                checkpoint_folder_path, f"{specific_ckpt_folder_name}/segmented.pkl"
            ),
        )

    labels_data = np.array(labels_data)

    return np.array(new_data), np.concatenate(labels_data)


def equalize_audio_duration(audio_datas: np.ndarray, offset: int = None) -> np.ndarray:
    if offset is None:
        # Get length of audio by mapping len function
        audio_length_func = np.vectorize(len)
        audio_length_arr = audio_length_func(audio_datas)

        # Set offset value by higher outlier
        first_quartile = np.percentile(audio_length_arr, 25)
        third_quartile = np.percentile(audio_length_arr, 75)
        iqr = third_quartile - first_quartile
        offset = round(third_quartile + 1.5 * iqr)

    new_audio_datas = []

    print(f"Equalizing data with offset={offset} samples...")

    for audio_data in tqdm(audio_datas, total=len(audio_datas)):
        # If length less than offset, then center pad it
        if len(audio_data) <= offset:
            padded_data = librosa.util.pad_center(data=audio_data, size=offset)
            new_audio_datas.append(padded_data.astype(np.float32))
            continue

        # Else, trim it
        # Randomize the start state
        random_start = np.random.randint(0, len(audio_data) - offset)

        # Trim it
        new_audio_datas.append(
            audio_data[random_start : (random_start + offset)].astype(np.float32)
        )

    print("Data equalized.")

    return np.array(new_audio_datas)


def generate_augmented_data(
    audio_datas: np.ndarray, n_aug: int, sampling_rate: int = 16000
) -> np.ndarray:
    # Precondition: all datas are from most discriminated data
    # ic(audio_datas.shape)

    # Check n_aug is defined
    if n_aug is None:
        raise Exception(
            "Please specify n_aug or how many data you want to be augmented."
        )

    augmented_datas = []

    # Gaussian Noise
    gaussian_noise = AddGaussianNoise()

    # Gain
    gain = Gain(min_gain_in_db=-6, max_gain_in_db=6)

    # Compose
    augment = Compose([gaussian_noise, gain], shuffle=True)

    print("Augmenting data...")

    for _ in tqdm(range(n_aug), total=n_aug):
        random_audio_data = audio_datas[np.random.choice(len(audio_datas))]
        # ic(random_audio_data)
        aug = augment(samples=random_audio_data, sample_rate=sampling_rate)
        augmented_datas.append(aug.astype(np.float32))

    return np.array(augmented_datas)


def extract_melspec(
    audio_datas: np.ndarray,
    sampling_rate: int = 16000,
    is_normalize: bool = True,
    in_db: bool = True,
    var_norm: bool = False,
    **kwargs,
) -> np.ndarray:
    # Initiate new container
    mel_specs = []

    print("Extracting mel spectrograms from audio...")

    # For each audio
    for audio in tqdm(audio_datas):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sampling_rate,
            window=signal.windows.hamming,
            win_length=512,
            n_fft=512,
            **kwargs,
        )

        if in_db:
            # Change to decibels instead of power spectrogram
            log_mel_spec = librosa.power_to_db(mel_spec)
        else:
            log_mel_spec = mel_spec

        if is_normalize:
            normalized = cmvnw(np.array(log_mel_spec), variance_normalization=var_norm)
            mel_specs.append(normalized)
        else:
            mel_specs.append(log_mel_spec)

    return np.array(mel_specs)
