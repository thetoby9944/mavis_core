import glob
import pkgutil
import sys
import os
import traceback
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from os.path import join, dirname, basename, isfile
from pathlib import Path, PureWindowsPath, PurePosixPath
from runpy import run_module
from zipfile import ZipFile, ZIP_STORED

import hydralit_components as hc
import pandas as pd
import streamlit as st
from PIL import Image
from natsort import os_sorted
from st_aggrid import GridOptionsBuilder, AgGrid

import config
import db
from db import ProjectDAO, current_data_dir, DFDAO, ConfigDAO
from db import config_path, LoginDAO, LocalFolderBrowserMixin
from pdutils import image_columns
from pdutils import overwrite_modes, fill_column
from pilutils import FILETYPE_EXTENSIONS, IMAGE_FILETYPE_EXTENSIONS
from ui.sessionstate import get
from v2.ui.processors.ppt import PPTExportWidget


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


class FileUpload:
    def __init__(
            self,
            target_dir,
            label="Upload Files",
            type=FILETYPE_EXTENSIONS,
            accept_multiple_files=True,
            help="The files will be uploaded to the mavis data path in the corresponding project directory. "
                 "You can set the mavis data root in the side panel under settings. "
    ):
        self.target_dir = Path(target_dir)
        self.accept_multiple_files = accept_multiple_files
        self.uploaded_files = st.file_uploader(label, type, accept_multiple_files, help=help)
        if not self.accept_multiple_files:
            self.uploaded_files = [self.uploaded_files]

    def start(self):
        self.target_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for file in self.uploaded_files:
            if file is None:
                st.warning("Upload a file")
                continue

            target_path = str((self.target_dir / file.name).resolve())
            suffix = Path(file.name).suffix.lower()

            if suffix in IMAGE_FILETYPE_EXTENSIONS:
                img = Image.open(file).convert("RGB")
                img.save(target_path)
                paths += [target_path]

            elif suffix in [".zip"]:
                target_dir = self.target_dir.resolve()  # / Path(file.name).stem).resolve()
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
        st.markdown("#### Import from Host")

        help = (
            "This pipeline allows to select images from the file system of the host system.\n"
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
            old_root_dir_selection = ConfigDAO()["root_dir"]
            root_dir_selection = empty_radio.radio("Go To", self.get_folders(ConfigDAO()["root_dir"]))
            if root_dir_selection != old_root_dir_selection:
                ConfigDAO()["root_dir"] = (ConfigDAO()["root_dir"] / root_dir_selection).resolve()
                root_dir_selection = empty_radio.radio("Go To", self.get_folders(ConfigDAO()["root_dir"]))

            old_root_dir_selection = ConfigDAO()["root_dir"]
            ConfigDAO()["root_dir"] = Path(st.text_input("Current Directory", ConfigDAO()["root_dir"]))
            # if ConfigDAO()["root_dir != old_root_dir_selection:
            #    root_dir_selection = empty_radio.selectbox("Go To", [ConfigDAO()["root_dir] + list(glob.glob(str(ConfigDAO()["root_dir / "**"))))

            if st.button("Use Current Directory"):
                ConfigDAO()["selection"] = str(ConfigDAO()["root_dir"] / "**")

        ConfigDAO()["selection"] = st.text_input("File selection expression", ConfigDAO()["selection"], help=help)

        ConfigDAO()["load_folders_as_columns"] = st.checkbox("Load matching folders as separate columns. "
                                                             "Uncheck to load all matching files in one column.",
                                                             ConfigDAO()["load_folders_as_columns"])

        if not ConfigDAO()["load_folders_as_columns"]:
            ConfigDAO()["column"] = st.text_input("Remember matched paths in:", ConfigDAO()["column"])

        config_min_file_size = 0.
        if st.checkbox("Show Extended options"):
            ConfigDAO()["is_sbi"] = not ConfigDAO()["load_folders_as_columns"] and st.checkbox("Folder is SBI",
                                                                                               ConfigDAO()["is_sbi"])
            ConfigDAO()["recursive"] = not ConfigDAO()["is_sbi"] and st.checkbox(
                "Collect files recursively. Uncheck to match exact expression.",
                ConfigDAO()["recursive"])
            ConfigDAO()["sort"] = st.checkbox("Keep file order sorted OS-style", ConfigDAO()["sort"])
            config_min_file_size = float(st.number_input("Minimum file size in KB. Zero to ignore.", 0.))
            ConfigDAO()["overwrite_modes"] = st.radio("When creating a column which is already present",
                                                      list(overwrite_modes.values()),
                                                      list(overwrite_modes.values()).index(
                                                          ConfigDAO()["overwrite_modes"]))

        if st.button("Load"):
            df = DFDAO().get(ProjectDAO().get())

            selection = ConfigDAO()["selection"]
            if ConfigDAO()["is_sbi"]:
                selection = str(Path(ConfigDAO()["selection"]) / "0" / "**" / "**" / "*.png")
            files = glob.glob(selection, recursive=ConfigDAO()["recursive"])

            if ConfigDAO()["sort"]:
                files = os_sorted(files)

            if ConfigDAO()["load_folders_as_columns"]:
                files_per_folder = defaultdict(list)
                for file in files:
                    if Path(file).is_file() and (config_min_file_size == 0
                                                 or config_min_file_size < Path(file).stat().st_size / 1024):
                        folder = Path(file).parent.stem
                        files_per_folder[folder] += [file]
                for folder in list(files_per_folder.keys()):
                    df = fill_column(df, folder, files_per_folder[folder], ConfigDAO()["overwrite_modes"])
                    st.info(f"Loaded {len(files_per_folder[folder])} paths from folder **{folder}**")
            else:
                df = fill_column(df, ConfigDAO()["column"], files, ConfigDAO()["overwrite_modes"])
                st.info(f"Loaded a total of {len(files)} paths.")
            DFDAO().set(df, ProjectDAO().get())

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
            overwrite_mode = st.radio("When creating a column which is already present",
                                      list(overwrite_modes.values()), 0)

        if st.button("Upload"):
            df = DFDAO().get(ProjectDAO().get())
            paths = uploader.start()
            if paths:
                df = fill_column(df, column, paths, overwrite_mode)
                DFDAO().set(df, ProjectDAO().get())


class UploadZipWidget:
    def __init__(self):
        st.markdown("#### Upload .zip")
        uploader = FileUpload(current_data_dir(), "Upload Archive", type=[".zip"], accept_multiple_files=False)

        if st.button("Extract Archive"):
            target_dir = uploader.start()
            if target_dir:
                st.warning(f"Use 'import from host' functionality to access desired files")
                ConfigDAO()["selection"] = str(target_dir)


class ImportHelperWidget:
    def __init__(self):
        st.markdown("#### Import Helper")
        ConfigDAO()["selection"] = st.text_input(
            "Select Directory to Analyze", ConfigDAO()["selection"],
            help="Uploads files when mavis is running on a remote server. "
                 "It will create a new folder under the project with the files."
        )
        selection = ConfigDAO()["selection"]
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

                    if ConfigDAO()["sort"]:
                        files = os_sorted(files)

                    df = fill_column(df, stem, [str(f) for f in files], ConfigDAO()["overwrite_modes"])
                    st.info(f"Loaded {len(files)} paths from folder **{stem}**")

            DFDAO().set(df, ProjectDAO().get())


class FileImportWidget:
    ConfigDAO()["selection"] = ConfigDAO(current_data_dir())["selection"]
    ConfigDAO()["recursive"] = ConfigDAO(False)["recursive"]
    ConfigDAO()["column"] = ConfigDAO("Images")["column"]
    ConfigDAO()["load_folders_as_columns"] = ConfigDAO(True)["load_folders_as_columns"]
    ConfigDAO()["sort"] = ConfigDAO(True)["sort"]
    ConfigDAO()["overwrite_modes"] = ConfigDAO(list(overwrite_modes.values())[0])["overwrite_modes"]
    ConfigDAO()["root_dir"] = ConfigDAO(Path.home())["root_dir"]
    ConfigDAO()["is_sbi"] = ConfigDAO(False)["is_sbi"]

    def __init__(self):
        folder_picker_column, upload_files_column, upload_zip_column = st.columns(3)
        with folder_picker_column:
            project = ProjectDAO().get()
            df = DFDAO.get(project)
            AddWidget(df, project)

        with upload_files_column:
            FileUploaderWidget()

        with upload_zip_column:
            UploadZipWidget()

        st.write("---")
        csv_upload_column, import_helper_column, regex_column = st.columns(3)
        with csv_upload_column:
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

        with import_helper_column:
            ImportHelperWidget()

        with regex_column:
            ImportFromHostWidget()


class SlimProjectWidget:
    def __init__(self):
        current = Path(ProjectDAO().get()).stem
        projects = os_sorted(ProjectDAO().get_all())
        selection = st.selectbox("Select Project", projects, projects.index(current) if current in projects else 0)
        if selection != current:
            ProjectDAO().set(selection)


class GalleryWidget:
    def __init__(self, df):
        col_selection, col_options = st.columns(2)

        project = ProjectDAO().get()

        with col_selection:
            columns = st.multiselect("⋮⋮⋮", image_columns(df), help="Show an Image Gallery")
        if columns:
            with col_options:

                n_columns = len(columns)
                st.write("")
                st.write("")
                display_options = st.expander("⋮⋮⋮ Options")

                with display_options:
                    columns_per_column = st.slider("⋮⋮⋮ Columns", 1, 15, max(5 - n_columns, 2))
                    n_row = columns_per_column * n_columns
                    max_items_per_page = st.slider(
                        "Images per Page",
                        1, 500, max(int((n_row ** 2) / n_columns * 0.75), 1)
                    )
                    show_caption = st.checkbox("Show Caption")
                    is_selectable = st.checkbox("Flag Images")
                    df_filtered = df

                    if st.checkbox("Filter"):
                        filter_column = st.selectbox("Select Filter Column", df.columns)
                        filter_values = st.multiselect("Filter By", list(df[filter_column].unique()))
                        if filter_values:
                            df_filtered = df[df[filter_column].isin(filter_values)]
                        df = df_filtered

                    order_column = st.multiselect("Order display by column", df.columns)
                    if order_column:
                        desc = st.checkbox("Descending", False, key="Desc.Img.Gallery")
                        df = df.sort_values(order_column, ascending=not desc).reset_index(drop=True)
                        st.info(f"Ordered by {order_column}")
                    if is_selectable:
                        flag_column = st.text_input("Flag Column", "Flag")
                        flag_value = st.text_input("Flag Value")

            with st.expander("⋮⋮⋮", True):
                paths = df[columns].dropna()
                items_per_page = (min(len(paths), max_items_per_page))
                n_pages = (len(paths) - 1) // items_per_page + 1
                page_number = (st.slider("Page", 1, n_pages) - 1) if n_pages > 1 else 0

                min_index = page_number * items_per_page
                max_index = min(min_index + items_per_page, len(paths))

                selections = {}
                current_ind = min_index
                while current_ind < max_index:
                    current_column = 0
                    ind_layout = st.columns(columns_per_column)
                    col_layout = st.columns(n_columns * columns_per_column)
                    for i in range(columns_per_column):

                        df_index = paths.index[current_ind]

                        if is_selectable:
                            with ind_layout[i]:
                                selections[df_index] = st.checkbox(f"{df_index}", flag_column in df and df[flag_column][
                                    df_index] == flag_value)

                        for j in range(n_columns):
                            with col_layout[current_column]:
                                col_name = columns[j]
                                path = paths[col_name][df_index]
                                caption = f'{df_index}: {Path(path).name}' if show_caption else ""
                                st.image(Image.open(path), use_column_width=True, caption=caption)
                                current_column += 1
                        current_ind += 1
                        if current_ind >= max_index:
                            break

                if st.button(".ppt"):
                    PPTExportWidget(paths[columns][min_index: max_index])

            if is_selectable:
                with display_options:
                    if flag_column not in df:
                        df[flag_column] = None
                    if st.button("Apply"):
                        for k, v in selections.items():
                            df.loc[k, flag_column] = flag_value if v else df.loc[k, flag_column]
                        DFDAO().set(df, project)
                        st.info("Updated Flags.")


class TableWidget:
    def __init__(self, df):
        if len(df) == 0:
            return
        table = st.empty()
        max_items = min(len(df), 10)
        omit_col, import_col, export_col = st.columns([10, 1, 1])
        with omit_col:
            if len(df) > max_items:
                max_items = len(df) if st.checkbox("Some rows have been ommited from "
                                                   "display due to performance. Check to view all") else max_items
        csv_args = {
            "sep": ";",
            "decimal": ",",
            "header": 1
        }
        name = Path(ProjectDAO().get()).stem

        with export_col:
            if st.button("Download .csv"):
                ExportWidget(f"{name}.csv").df_link(csv_args)

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
        with st.spinner("Loading Table"):
            with table:
                AgGrid(
                    temp_df,
                    # update_mode=GridUpdateMode.MANUAL,
                    gridOptions=grid.build(),
                    fit_columns_on_grid_load=True,
                    allow_unsafe_jscode=True,
                    height=200
                )


class LoginWidget:
    def check(self):
        login_column = st.empty()
        username, password = get(username="default", password="")
        result, username, password = LoginDAO().check_session(username, password)
        if not result:
            column = login_column.columns(3)[1]

            form = column.form("Login")
            with form:
                st.write("### Login")
                username = st.text_input("Username:", key="userlogin")
                password = st.text_input("Password:", type="password", key="userpw")
                if st.form_submit_button("Login"):

                    result, username, password = LoginDAO().check_session(username, password)
                    if password and not result:
                        st.warning("Please enter valid credentials")

                    if result:
                        st.success(f"Logged in as: {username}")
                        st.session_state.username = username
                        st.session_state.password = password
                        login_column.empty()
                        st.experimental_rerun()
        return result


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


class BodyWidget:
    def __init__(self):
        try:
            project = ProjectDAO().get()
            st.write(f"# 🗀  {Path(ProjectDAO().get()).stem}")

            col1, col2, col3, col4 = st.columns([2, 1, 2, 1])
            with col1:
                SlimProjectWidget()
            with col2:
                st.write("  ")
                st.write("  ")
                if st.button(" Reload Project"):
                    DFDAO().get(None)
            with col3:
                name = st.text_input("New Project Name", f"{datetime.now():%y%m%d}")
            with col4:
                st.write("  ")
                st.write("  ")
                if st.button("🞥 Create Project"):
                    ProjectDAO().add(name)

            df = DFDAO().get(project)
            TableWidget(df)

            if df.empty:
                cols = st.columns(3)
                with cols[1]:
                    st.image("assets/images/wowsuchemtpy.png")
                    st.write("### Wow, such empty")
                    st.info("Start by uploading images to **🗀** the project, `\n` or use the **`Project`** module to "
                            "import .zip archives or locate files on the host system.")

            with st.expander("🞥 Add Files"):
                FileImportWidget()

            st.write("  \n")

            if len(image_columns(df)):
                GalleryWidget(df)


        except:
            st.error(
                "Display raised a message. "
                "Try **'R'** to rerun"
            )
            st.button("Refresh")
            with st.expander("Message"):
                st.code(traceback.format_exc())


class ModuleWidget:
    def __init__(self):
        # Packages
        package_path = db.ModulePathDAO().get() or "pipelines"
        # print(f"Looking for mavis modules in {package_path}")
        if str(package_path) not in sys.path:
            sys.path.append(str(package_path))

        self.packages = [
            __import__(modname, fromlist="dummy")
            for importer, modname, ispkg in pkgutil.iter_modules([str(package_path)], "")
        ]
        self.pack_names = [pack.__name__.split('.')[-1] for pack in self.packages]

        # Licenses
        self.license_path = str(Path(config_path()).parent / "license_keys.txt")
        try:
            with open(self.license_path) as f:
                self.licenses = f.read()
        except:
            self.licenses = []

    def get_modules(self):
        for pack, pack_name in zip(self.packages, self.pack_names):

            modules = glob.glob(join(dirname(pack.__file__), "*.py"))
            modules = [basename(f)[:-3] for f in modules if isfile(f)
                       and not f.endswith('__init__.py')]

            for module_name in sorted(modules):
                full_module_name = f"{pack.__name__}.{module_name}"
                yield full_module_name

    def get_modules_interactive(self):
        search = st.sidebar.text_input(f"What do you want to do?")
        # st.write("---")
        # uploader = FileUpload(Path(self.license_path).parent, "Upload a License file", ".txt", False)
        # if st.button("Upload License"):
        #     target_dir = uploader.start()
        #     if target_dir:
        #         st.success("Uploaded a license file. Press **`R`** to refresh.")

        for pack, pack_name in zip(self.packages, self.pack_names):
            # if pack_name not in self.licenses:
            #     st.sidebar.warning(f"Package {pack_name} is not licensed.")
            #     continue
            #     pass

            # expander = st.sidebar.expander(pack_name, expanded=bool(search))

            modules = glob.glob(join(dirname(pack.__file__), "*.py"))
            modules = [basename(f)[:-3] for f in modules if isfile(f)
                       and not f.endswith('__init__.py')]

            for module_name in sorted(modules):
                if search and search.lower() not in module_name.lower():
                    continue

                full_module_name = f"{pack.__name__}.{module_name}"

                yield full_module_name, module_name, pack_name

    def execute(self):

        modules_by_pack = defaultdict(list)
        modules_by_id = {}
        for full_module_name, module_name, pack_name in self.get_modules_interactive():
            modules_by_pack[pack_name] += [(full_module_name, module_name, pack_name)]
            modules_by_id[full_module_name] = [module_name, pack_name]

        menu_data = [
            {
                'label': pack_name,
                'submenu': [
                    {
                        'label': module_name,
                        "id": full_module_name

                    }
                    for full_module_name, module_name, pack_name
                    in pack
                ]

            }
            for pack_name, pack
            in modules_by_pack.items()
        ]

        over_theme = {'txc_inactive': '#FFFFFF'}
        over_theme = {
            'txc_inactive': '#000000',
            'menu_background': '#FFFFFF',
            'txc_active': 'lightblue',
            # 'option_active':'blue'
        }

        menu_id = hc.nav_bar(menu_definition=menu_data, override_theme=over_theme, sticky_nav=True, home_name="Home")

        if menu_id in modules_by_id:
            # if "centered" != st.session_state.layout:
            #    st.session_state.layout = "centered"
            #    st.experimental_rerun()

            name, full_name = modules_by_id[menu_id]
            db.BaseDAO.ACTIVE_PIPELINE = name
            st.markdown(f"# {name}")
            run_module(menu_id)

        if menu_id == "Home":
            # if "wide" != st.session_state.layout:
            #    st.session_state.layout = "wide"
            #    st.experimental_rerun()

            BodyWidget()

        with st.sidebar:
            package_path = db.ModulePathDAO().get()

            st.write("---")
            with st.expander("🞥 Add functionality"):
                upload_widget = st.empty()
                with upload_widget:
                    uploader = FileUpload(
                        str(package_path), f"Upload package", ".zip",
                        help="Add functionality by uploading zipped python packages. "
                        # "The python packages can contain arbitrary python scripts."
                        # "After uploading a package, all its contents will be added to the menu."
                        # "Clicking on a script in the menu will import and execute that script. "
                        # "Hence, preferably you add stremalit scripts that run on import."
                    )

                if st.button(f"Upload package"):
                    uploader.start()

            # st.write("---")
            with st.expander("⚙ Settings"):
                db.ModulePathDAO().edit_widget()
                db.LogPathDAO().edit_widget()
                db.DataPathDAO().edit_widget()

                st.write("---")
                st.write("### Reset")
                if st.button(
                        f"Reset Presets",
                        help=f"Resets {db.BaseDAO.ACTIVE_PIPELINE}"
                ):
                    config.PresetListDAO().reset()
                    config.ActivePresetDAO().reset()
                    config.ModelDAO().reset()
                    config.ConfigDAO().reset()

                st.write("---")
                st.write("### Versions")

                from mavis import __version__
                import tensorflow

                versions = pd.DataFrame([
                    {f"Version": f"{__version__}"},
                    {f"Version": f"{tensorflow.__version__}"}
                ], index=["Mavis", "Tensorflow"])

                st.write(versions)


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
