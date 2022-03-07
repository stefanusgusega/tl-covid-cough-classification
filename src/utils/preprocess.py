from typing import Tuple
import numpy as np
import pandas as pd
import pickle as pkl
from utils.audio import convert_audio_to_numpy, generate_segmented_data


def preprocess_dataframe(
    df: pd.DataFrame,
    audio_folder_path: str,
    sampling_rate: int = 16000,
    save_to_pickle=True,
    path_dump=None,
) -> Tuple[np.ndarray, np.ndarray]:
    numpy_data, covid_statuses = convert_audio_to_numpy(
        df, audio_folder_path=audio_folder_path, sampling_rate=sampling_rate
    )

    segmented_data, segmented_covid_status = generate_segmented_data(
        numpy_data, covid_statuses, sampling_rate=sampling_rate
    )

    # TO-DO: feature extraction and check segmentation

    res = segmented_data, segmented_covid_status

    # Save to pickle file
    if save_to_pickle:
        # Check if path_dump is specified, if not so raise an exception
        if path_dump is None:
            raise Exception(
                "Please specify the path that you wish to dump the result of this conversion."
            )
        # Then dump it
        with (open(path_dump, "wb")) as f:
            pkl.dump(res, f)

    # Returning series of data in (-1, 1) shape and the labels in (-1, 1) too
    # NOW : segmented_data in (-1,) shape
    return res
