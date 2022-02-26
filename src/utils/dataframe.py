import os
import pandas as pd


def append_format(
    dataset_path: str,
    df: pd.DataFrame,
    formats: list,
    filename_column: str,
    inplace=False,
):
    ext_arr = []

    for _, row in df.iterrows():
        ext = ""
        found = False
        i = 0
        while not found and i < len(formats):
            if os.path.exists(
                os.path.join(dataset_path, row[filename_column] + formats[i])
            ):
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


def filter_cough(df: pd.DataFrame, column_name: str, threshold_detected: float = 0.9):
    if column_name not in df.columns:
        raise Exception(f"No column named '{column_name}'")

    return df[df[column_name] >= threshold_detected]


def filter_covid(
    df: pd.DataFrame,
    column_name: str,
    pos_label: str = "COVID-19",
    neg_label: str = "healthy",
):
    if column_name not in df.columns:
        raise Exception(f"No column named '{column_name}'")

    if pos_label not in list(df[column_name].unique()):
        raise Exception(f"No label named '{pos_label}'")

    if neg_label not in list(df[column_name].unique()):
        raise Exception(f"No label named '{neg_label}'")

    return df[(df[column_name] == pos_label) | (df[column_name] == neg_label)]
