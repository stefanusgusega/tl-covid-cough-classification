import os
import string
import pandas as pd
import numpy as np
import librosa


def segment_cough(
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
    coughSegments = []
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
                    if cough_end + 1 - cough_start - 2 * padding > min_cough_samples:
                        coughSegments.append(x[cough_start : cough_end + 1])
                        cough_mask[cough_start : cough_end + 1] = True
            elif i == (len(x) - 1):
                cough_end = i
                cough_in_progress = False
                if cough_end + 1 - cough_start - 2 * padding > min_cough_samples:
                    coughSegments.append(x[cough_start : cough_end + 1])
            else:
                below_th_counter = 0
        else:
            if sample > seg_th_h:
                cough_start = i - padding if (i - padding >= 0) else 0
                cough_in_progress = True

    return coughSegments, cough_mask


def append_format(dataset_path: str, df: pd.DataFrame, formats: list, inplace=False):
    ext_arr = []

    for _, row in df.iterrows():
        ext = ""
        found = False
        i = 0
        while not found and i < len(formats):
            if os.path.exists(os.path.join(dataset_path, row["uuid"] + formats[i])):
                ext = formats[i]
                found = True

            i += 1

        if ext == "":
            raise Exception("No audio file found.")

        ext_arr.append(ext)

    with_ext_df = pd.concat([df, pd.DataFrame({"ext": ext_arr})], axis=1)

    if inplace:
        df = with_ext_df

    return with_ext_df
