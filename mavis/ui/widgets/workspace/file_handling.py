import glob
import os
import sys
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from pathlib import Path, PureWindowsPath, PurePosixPath
from zipfile import ZipFile, ZIP_STORED

import pandas as pd
import streamlit as st
from PIL import Image
from natsort import os_sorted
from streamlit_option_menu import option_menu

from mavis.db import DFDAO, ProjectDAO, current_data_dir, LocalFolderBrowserMixin
from mavis.pdutils import overwrite_modes, fill_column
from mavis.pilutils import FILETYPE_EXTENSIONS, IMAGE_FILETYPE_EXTENSIONS
from mavis.presets.base import BaseConfig


class FileSettings(BaseConfig):
    def parameter_block(self):
        pass

    selection = current_data_dir()
    recursive = False
    column = "Images"
    load_folders_as_columns = True
    sort = True
    overwrite_modes = list(overwrite_modes.values())[0]
    root_dir = Path.home()
    is_sbi = False


class FileUpload:
    def __init__(
            self,
            target_dir,
            label="Upload Files",
            type=FILETYPE_EXTENSIONS,
            accept_multiple_files=True,
            help="The files will be uploaded to the mavis data path in the corresponding project directory. "
                 "You can set the mavis data root in the side panel under settings. ",
            landing_zone_copy=False
    ):
        self.target_dir = Path(target_dir)
        self.accept_multiple_files = accept_multiple_files
        self.uploaded_files = st.file_uploader(label, type, accept_multiple_files, help=help)
        self.landing_zone_copy = landing_zone_copy
        if not self.accept_multiple_files:
            self.uploaded_files = [self.uploaded_files]

    def start(self):
        self.target_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for file in self.uploaded_files:
            if file is None:
                st.warning("No file selected.")
                continue

            target_path = str((self.target_dir / Path(file.name)).resolve())
            suffix = Path(file.name).suffix.lower()

            if self.landing_zone_copy:
                date_str = f'{datetime.now().strftime("%Y%m%d")}'
                landing_path = str(
                    (Path(current_data_dir()) / "landing_zone" / date_str / file.name).resolve()
                )
                with open(landing_path, "wb") as target:
                    target.write(file.read())

            if suffix in IMAGE_FILETYPE_EXTENSIONS:
                img = Image.open(file).convert("RGB")
                img.save(target_path)
                paths += [target_path]

            elif suffix in [".zip"]:
                target_dir = (self.target_dir / Path(file.name).stem).resolve()
                target_dir.mkdir(parents=True, exist_ok=True)
                ZipFile(file).extractall(target_dir)
                st.info(f"Extracted Archive to host into:")
                st.code(f"{target_dir}")
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


class ImportFromHostWidget:

    @staticmethod
    def get_folders(directory: Path):
        return [directory, ".."] + os_sorted([path for path in glob.glob(str(directory / "**")) if Path(path).is_dir()])

    def __init__(self):
        c = FileSettings()
        st.markdown("#### Import from Host")

        help = (
            "This module allows to select images from the file system of the host system.\n"
            "Upload the images to the host machine or a network drive which is available there.\n"
            "You can use WinSCP, the explorer, or FTP and then select the images here by providing a path.\n"
            "TIPS: "
            " - You can paste the exact path under windows by clicking SHIFT-Right Click and select:  \n"
            "   `Copy as path`\n"
            " - Select multiple images at once by using a placeholder `*`\n"
            " - You can use `regular expression` to find all desired files on your system."
        )
        if st.checkbox("List and browse directory"):
            empty_radio = st.empty()
            old_root_dir_selection = c.root_dir
            root_dir_selection = empty_radio.radio("Go To", self.get_folders(c.root_dir))
            if root_dir_selection != old_root_dir_selection:
                c.root_dir = (c.root_dir / root_dir_selection).resolve()
                root_dir_selection = empty_radio.radio("Go To", self.get_folders(c.root_dir))

            old_root_dir_selection = c.root_dir
            c.root_dir = Path(st.text_input("Current Directory", c.root_dir))
            # if ConfigDAO()["root_dir != old_root_dir_selection:
            #    root_dir_selection = empty_radio.selectbox("Go To", [ConfigDAO()["root_dir] + list(glob.glob(str(ConfigDAO()["root_dir / "**"))))

            if st.button("Use Current Directory"):
                c.selection = str(c.root_dir / "**")

        c.selection = st.text_input("File selection expression", c.selection, help=help)

        c.load_folders_as_columns = st.checkbox(
            "Load matching folders as separate columns. "
            "Uncheck to load all matching files in one column.",
            c.load_folders_as_columns
        )

        if not c.load_folders_as_columns:
            c.column = st.text_input("Remember matched paths in:", c.column)

        config_min_file_size = 0.
        if st.checkbox("Show Extended options"):
            c.is_sbi = not c.load_folders_as_columns and st.checkbox("Folder is SBI", c.is_sbi)
            c.recursive = not c.is_sbi and st.checkbox(
                "Collect files recursively. Uncheck to match exact expression.",
                c.recursive
            )
            c.sort = st.checkbox("Keep file order sorted OS-style", c.sort)
            config_min_file_size = float(st.number_input("Minimum file size in KB. Zero to ignore.", 0.))
            c.overwrite_modes = st.radio(
                "When creating a column which is already present",
                list(overwrite_modes.values()),
                list(overwrite_modes.values()).index(c.overwrite_modes)
            )

        if st.button("Load"):
            df = DFDAO().get(ProjectDAO().get())

            selection = c.selection
            if c.is_sbi:
                selection = str(Path(c.selection) / "0" / "**" / "**" / "*.png")
            files = glob.glob(selection, recursive=c.recursive)

            if c.sort:
                files = os_sorted(files)

            if c.load_folders_as_columns:
                files_per_folder = defaultdict(list)
                for file in files:
                    if Path(file).is_file() and (config_min_file_size == 0
                                                 or config_min_file_size < Path(file).stat().st_size / 1024):
                        folder = Path(file).parent.stem
                        files_per_folder[folder] += [file]
                for folder in list(files_per_folder.keys()):
                    df = fill_column(df, folder, files_per_folder[folder], c.overwrite_modes)
                    st.info(f"Loaded {len(files_per_folder[folder])} paths from folder **{folder}**")
            else:
                df = fill_column(df, c.column, files, c.overwrite_modes)
                st.info(f"Loaded a total of {len(files)} paths.")
            DFDAO().set(df, ProjectDAO().get())
        c.update()
        # st.write(DFDAO().get(ProjectDAO().get()))


class FileUploaderWidget:
    def __init__(self, verbose=False):
        st.write("#### Upload files")

        # st.markdown("### Upload Files")
        column = st.text_input(
            "Upload files. New Folder Name:", "Images",
            help="Uploads files when mavis is running on a remote server. "
                 "It will create a new folder under the project with the files."
        )
        uploader = FileUpload(Path(current_data_dir()) / column)
        overwrite_mode = overwrite_modes["opt_a"]
        if verbose:
            overwrite_mode = st.radio(
                "When creating a column which is already present",
                list(overwrite_modes.values()),
                0
            )

        if st.button("Upload"):
            df = DFDAO().get(ProjectDAO().get())
            paths = uploader.start()
            if paths:
                df = fill_column(df, column, paths, overwrite_mode)
                DFDAO().set(df, ProjectDAO().get())


class UploadZipWidget:
    def __init__(self):
        c = FileSettings()
        target_dir = str(Path(current_data_dir()) / "processing")
        st.markdown("#### Upload .zip")
        landing_zone_copy = st.checkbox(
            "Save a copy of the archive to the landing zone with current date.",
            value=True
        )
        uploader = FileUpload(
            target_dir=target_dir,
            label="Upload Archive",
            type=[".zip"],
            accept_multiple_files=False,
            landing_zone_copy=landing_zone_copy
        )

        if st.button("Extract Archive"):
            target_dir = uploader.start()
            if target_dir:
                st.warning(
                    f"Successfully extracted archive. "
                    f"Use `Locate on host` functionality to access desired files"
                )
                c.selection = str(target_dir)
                c.update()


class ImportHelperWidget:
    def __init__(self):
        c = FileSettings()
        st.markdown("#### Import Helper")
        c.selection = st.text_input(
            "Select Directory to Analyze", c.selection,
            help="Uploads files when mavis is running on a remote server. "
                 "It will create a new folder under the project with the files."
        )
        selection = c.selection
        folders, sub_dirs, file_types_list = [], [], []
        preview = st.checkbox("Preview")
        resolve_existing = st.checkbox("Try to resolve existing columns", True)
        st.write("###### Detected Folders")
        for i, (folder, sub_dir, files) in enumerate(os.walk(selection)):
            file_types = {Path(file).suffix.lower() for file in files}
            stem = Path(folder).stem
            st.write(f"Folder `{stem}`: {len(files)} Files {(f'of type {file_types}' if file_types else '')}")
            image_file_types = set(IMAGE_FILETYPE_EXTENSIONS).intersection(file_types)

            if len(image_file_types) > 0 and preview:
                st.image(Image.open(next(Path(folder).glob(f"*{list(image_file_types)[0]}"))), width=100)
            if file_types and st.checkbox(f"Import `{stem}`", key=f"{i}_{stem}"):
                folders.append(folder)
                if len(file_types) > 1:
                    selected_file_types = st.multiselect("Select file types to import", list(file_types))
                else:
                    selected_file_types = list(file_types)
                file_types_list.append(selected_file_types)

        if st.button("Import Paths"):
            df = DFDAO().get(ProjectDAO().get())
            for folder, file_types in zip(folders, file_types_list):
                stem = Path(folder).stem
                if resolve_existing and stem in df.columns:
                    df[f"{stem} Resolved"] = df[stem].dropna().apply(
                        lambda x: str(Path(selection) / (PureWindowsPath(x) if "\\" in str(x) else PurePosixPath(x)))
                    )
                    st.info(f"Resolved column **{stem}**")

                else:
                    files = []
                    for file_type in file_types:
                        files.extend((Path(folder).glob(f"*{file_type}")))

                    if c.sort:
                        files = os_sorted(files)

                    df = fill_column(df, stem, [str(f) for f in files], c.overwrite_modes)
                    st.info(f"Loaded {len(files)} paths from folder **{stem}**")

            DFDAO().set(df, ProjectDAO().get())


class FileImportWidget:

    def __init__(self):
        upload_files_column, folder_picker_column, upload_zip_column, \
        csv_upload_column, import_helper_column, regex_column = all_options = [
            "Upload Files",
            "Native Folder Browser (localhost only)",
            "Upload & Extract .zip",
            "Upload CSV File as Project Table",
            "Locate on host",
            "Locate by RegEx"
        ]

        selected = option_menu(
            "File Import Method",
            all_options,
            menu_icon="upload",
            icons=[
                "images",
                "search",
                "file-earmark-zip",
                "file-earmark-spreadsheet",
                "folder",
                "braces"
            ]
        )

        if selected == folder_picker_column and sys.platform.startswith('win'):
            project = ProjectDAO().get()
            df = DFDAO.get(project)
            AddWidget(df, project)

        if selected == upload_files_column:
            FileUploaderWidget()

        if selected == upload_zip_column:
            UploadZipWidget()

        if selected == csv_upload_column:
            st.write("#### Upload .csv")
            csv_args = {
                "sep": st.text_input("Column Sep.", ";"),
                "decimal": st.text_input("Decimal Sep", ","),
                "header": 0
            }
            name = Path(ProjectDAO().get()).stem
            uploaded_file = st.file_uploader(
                "",
                type=".csv",
                help="Upload .csv which contains any information or paths you want to process with Mavis.  \n"
                     "** WARNING: THIS WILL OVERWRITE THE CURRENT PROJECT **  \n"
                     "Default encoding of .csv is:  \n"
                     "- Decimal separator `,` \n"
                     "- Column separator `;`  \n"
                     "- header: `1`"
            )
            if uploaded_file is not None and st.button("⭱ .csv"):
                ProjectDAO().add(name, overwrite=True)
                DFDAO().set(pd.read_csv(uploaded_file, **csv_args), name)

        if selected == import_helper_column:
            ImportHelperWidget()

        if selected == regex_column:
            ImportFromHostWidget()


class AddWidget:
    def __init__(self, df, project):
        st.write("#### Localhost Browser")
        st.write("  ")
        # local = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sys.argv[0]).parent.resolve()
        local = LocalFolderBrowserMixin().browse()
        if local is not None and local:
            file_list = [
                str(local_file)
                for local_file in
                list(local.glob("*"))
                if local_file.is_file()
            ]

            df2 = fill_column(
                df,
                f"{local.name} ({local.parent.name})",
                file_list,
                overwrite_modes["opt_w"]
            )
            DFDAO().set(df2, project)


class ExportWidget:
    def __init__(self, name):
        self.name = name

    def _zip_dir(self, source_dirs, folder_names, pattern="*", verbose=True, recursive=False):
        target_file = BytesIO()
        with ZipFile(target_file, 'w', ZIP_STORED) as zf:
            for source_dir, folder_name in zip(source_dirs, folder_names):
                src_path = Path(source_dir).expanduser().resolve(strict=True)
                files = list(src_path.rglob(pattern) if recursive else src_path.glob(pattern))
                if verbose:
                    bar = st.progress(0)
                for i, file in enumerate(files):
                    if verbose:
                        bar.progress(i / len(files))
                    zf.write(file, Path(folder_name) / file.relative_to(src_path))
        target_file.seek(0)
        st.download_button(
            label=f"Download {self.name}",
            data=target_file,
            file_name=self.name
        )

    def df_link(self, csv_args):
        csv_args["header"] = True
        csv = DFDAO().get(ProjectDAO().get()).to_csv(index=False, **csv_args)
        st.download_button(
            label=f"Download",
            data=csv,
            file_name=self.name
        )

    def ds_link(self, paths, folder_names, recursive=False):
        self._zip_dir(paths, folder_names, recursive=recursive)
