import shelve
from pathlib import Path

import pandas as pd
import streamlit as st

from mavis.pathlibutils import maybe_new_dir
from mavis.pilutils import IMAGE_FILETYPE_EXTENSIONS
from mavis.shelveutils import load_df

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


def check_len(df1, df2):
    if len(df1.dropna(how="all")) <= len(df2.dropna(how="all")):
        return True
    else:
        st.warning("Will not update table. Updating would cause data loss because the new table contains less data.")
        return False


def save_df(df, base_path, new_dir, suffix, dir_name=None):
    p = Path(base_path)
    path = str(maybe_new_dir(p, dir_name, new_dir) / (p.stem + suffix))
    df.to_csv(path, sep=";", decimal=",")
    return path


def update(df, project):
    with shelve.open(project) as d:
        d["df"] = df
    load_df(None)
    load_df(project)


def file_columns(df, filter_list):
    x = [col for col in df.columns
            if any([t in Path(str(df[col][df[col].first_valid_index()])).suffix.lower()
                    for t in filter_list if df[col].first_valid_index() is not None])]
    return x[::-1]


def image_columns(df):
    return file_columns(df, IMAGE_FILETYPE_EXTENSIONS)


def document_columns(df):
    return file_columns(df, [".xml", ".json"])


def image_and_doc_columns(df):
    return document_columns(df) + image_columns(df)
