from pathlib import Path

import pandas as pd

from mavis.pilutils import IMAGE_FILETYPE_EXTENSIONS

overwrite_modes = {
    "opt_w": "Remove Existing Values (Overwrite)",
    "opt_a+": "Append to Existing Values",
    "opt_a": "Append to Existing Values (Drop Nans first)"
}


def fill_column(df, col, values, overwrite_mode=overwrite_modes["opt_a"]):
    df2 = pd.DataFrame({col: values})
    if overwrite_mode == overwrite_modes["opt_w"] and col in df.columns:
        df = df.drop([col], axis=1)
        df = pd.concat([df, df2], axis=1)

    elif overwrite_mode == overwrite_modes["opt_a+"] and col in df.columns:
        df = df.append(df2, ignore_index=True)

    elif overwrite_mode == overwrite_modes["opt_a"] and col in df.columns:
        df2 = pd.DataFrame({col: list(df[col].dropna()) + values})
        df = df.drop([col], axis=1)
        df = pd.concat([df, df2], axis=1)
    else:
        df = pd.concat([df, df2], axis=1)
    df = df.dropna(axis=0, how="all")
    return df


def file_columns(df, filter_list):
    x = [
        col for col in df.columns
        if any([
            t in Path(str(df[col][df[col].first_valid_index()])).suffix.lower()
            for t in filter_list if df[col].first_valid_index() is not None
        ])
    ]
    return x[::-1]


def image_columns(df):
    return file_columns(df, IMAGE_FILETYPE_EXTENSIONS)


def document_columns(df):
    return file_columns(df, [".xml", ".json"])


def image_and_doc_columns(df):
    return document_columns(df) + image_columns(df)


def numeric_columns(df: pd.DataFrame):
    return list(df.select_dtypes('number').columns.values)[::-1]
