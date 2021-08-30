import streamlit as st
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED


def zip_dir(target_file, source_dirs, folder_names, recursive=False, verbose=True):
    with ZipFile(target_file, 'w', ZIP_STORED) as zf:
        for source_dir, folder_name in zip(source_dirs, folder_names):
            src_path = Path(source_dir).expanduser().resolve(strict=True)
            files = list(src_path.rglob('*') if recursive else src_path.glob("*"))
            if verbose:
                bar = st.progress(0)
            for i, file in enumerate(files):
                if verbose:
                    bar.progress(i / len(files))
                zf.write(file, Path(folder_name) / file.relative_to(src_path))
