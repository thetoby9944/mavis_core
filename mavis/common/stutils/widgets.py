import pathlib
import traceback
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import streamlit as st
from PIL import Image
from st_aggrid import AgGrid, GridOptionsBuilder

from mavis.pdutils import image_columns
from mavis.pilutils import FILETYPE_EXTENSIONS, IMAGE_FILETYPE_EXTENSIONS
from mavis.shelveutils import current_project, load_df
from mavis.stutils.startpage.gallery import gallery
from mavis.stutils.startpage.toolbar import toolbar

# HACK This only works when we've installed streamlit with pipenv, so the
# permissions during install are the same as the running process
STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
# We create a downloads directory within the streamlit static asset directory
# and we write output files to it
DOWNLOADS_PATH = (STREAMLIT_STATIC_PATH / "downloads")
DOWNLOADS_PATH.mkdir(exist_ok=True)


def remote_css(url=""):
    url = ('https://fonts.googleapis.com/icon?family=Material+Icons') if not url else url
    st.markdown(
        f'<link href="{url}" rel="stylesheet">',
        unsafe_allow_html=True
    )


# remote_css()

def icon(icon_name):
    st.markdown(f'<span class="material-icons" style="color:blue;">{icon_name}</span>', unsafe_allow_html=True)


def rgb_picker(label, value):
    hex_str = st.color_picker(label, value)
    return tuple([int(hex_str.lstrip("#")[i:i + 2], 16)
                  for i in (0, 2, 4)])


def identifier_options(sample_path):
    parts = Path(sample_path).parts
    return [parts.index(sel)
            for sel in st.multiselect("Include names of parent directories:",
                                      parts)]


def contour_filter_options():
    min_circularity = st.number_input("Minimum circularity for detected segments", 0., 1., 0.)
    min_area = st.number_input("Minimum Area for detected segments. Set to zero to disable.", 0, 100000, 0)
    max_area = st.number_input("Maximum Area for detected segments. Set to zero to disable.", 0, 100000, 0)
    ignore_border_cnt = st.checkbox("Ignore Border Contours", True)
    return min_circularity, min_area, max_area, ignore_border_cnt


def header(project_path):
    try:

        df = load_df(project_path)

        with st.beta_expander(f"🗀  {Path(current_project()).stem}"):
            button = st.button(u"\u21BB")
            if button:
                load_df(None)

            table = st.empty()
            max_items = min(len(df), 100)
            if len(df) > 100:
                max_items = len(df) if st.checkbox("Some rows have been ommited from "
                                                   "display due to performance. Check to view all") else max_items
            temp_df = df.head(max_items)[df.columns[::-1]].round(2)
            pd.set_option('display.max_colwidth', None)
            pd.set_option("display.precision", 2)
            grid = GridOptionsBuilder().from_dataframe(temp_df)

            grid.configure_default_column(
                min_column_width=300,
                width=400,
                defaultWidth=400,
                # maxWidth=300,
                groupable=True,
                resizeable=True,
                dndSource=False,
                editable=True,
                cellStyle={"direction": "rtl", "text-overflow": "ellipsis"}
            )
            with table:

                AgGrid(temp_df, gridOptions=grid.build(), fit_columns_on_grid_load=True, allow_unsafe_jscode=True)

            # table.dataframe(df.head(max_items)[df.columns[::-1]])

        toolbar()

        if len(image_columns(df)):
            gallery()
        if not len(df):
            st.info("Start by locating or uploading files with: Files \\ Import Files")

    except:
        st.error("Cannot display Details")
        st.code(traceback.format_exc())


# def zip_link(zip_file: io.BytesIO, file_name):
#    zip_file.seek(0)
#    st.info("Encoding .zip")
#    b64 = base64.b64encode(zip_file.read()).decode()
#    st.info("Writing Link Content")
#    href = f'<a href="data:file/zip;base64,{b64}" download=\'{file_name}\'> Click to download {file_name}</a>'
#    st.markdown(href, unsafe_allow_html=True)


class FileUpload:
    def __init__(self, target_dir, label="Upload Files", type=FILETYPE_EXTENSIONS, accept_multiple_files=True):
        self.target_dir = Path(target_dir)
        self.accept_multiple_files = accept_multiple_files
        self.uploaded_files = st.file_uploader(label, type, accept_multiple_files)

        if not self.accept_multiple_files:
            self.uploaded_files = [self.uploaded_files]

    def start(self):
        self.target_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for file in self.uploaded_files:
            if file is None:
                st.warning("Upload a file")
                continue

            target_path = (self.target_dir / file.name).resolve()
            suffix = Path(file.name).suffix.lower()

            if suffix in IMAGE_FILETYPE_EXTENSIONS:
                img = Image.open(file).convert("RGB")
                img.save(target_path)
                paths += [target_path]

            elif suffix in [".zip"]:
                target_dir = (self.target_dir / Path(file.name).stem).resolve()
                target_dir.mkdir(parents=True, exist_ok=True)
                ZipFile(file).extractall(target_dir)
                st.info(f"Extracted Archive to host into {target_dir}")
                paths += [target_dir]

            else:
                with open(target_path, "wb") as target:
                    target.write(file.read())
                paths += [target_path]

        st.info(f"Uploaded {len(paths)} file(s).")
        if self.accept_multiple_files:
            return paths
        else:
            return next(iter(paths), None)
