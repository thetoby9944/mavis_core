import glob
import pkgutil
import sys
import traceback
from os.path import join, dirname, basename, isfile
from pathlib import Path
from runpy import run_module
from zipfile import ZipFile

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from natsort import os_sorted
from st_aggrid import GridOptionsBuilder, AgGrid

import config
import shelveutils
from config import ExportWidget
from pdutils import overwrite_modes, fill_column, image_columns
from pilutils import FILETYPE_EXTENSIONS, IMAGE_FILETYPE_EXTENSIONS
from shelveutils import ProjectDAO, DFDAO, config_path, LoginDAO, LocalFolderBrowserMixin


# import mavis
from stutils.sessionstate import get


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
    def __init__(self, target_dir, label="Upload Files", type=FILETYPE_EXTENSIONS,
                 accept_multiple_files=True):
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

            target_path = str((self.target_dir / file.name).resolve())
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


class EditWidget:
    def __init__(self, df):
        project = ProjectDAO().get()

        if not df.empty:
            columns_to_remove = st.multiselect("Remove column(s)", list(df.columns))
            delete_files = False  # st.checkbox("Delete associated files.")
            if st.button("Remove"):
                for column in columns_to_remove:
                    if delete_files:
                        for path in df[column]:
                            try:
                                path = Path(path)
                                if path.is_file():
                                    path.unlink()
                            except:
                                pass
                    df = df.drop(column, axis=1)
                    st.info(column + " removed")
                DFDAO().set(df, project, allow_loss=True)
            st.write("---")
            if not st.checkbox("More", False, key="showmoreedit"):
                return

            st.markdown("--- \n ### Duplicate a column")
            column = st.selectbox("Duplicate column", df.columns)
            name = st.text_input("New Column Name")
            if st.button("Duplicate"):
                df[name] = df[column]
                DFDAO().set(df, project)
                st.info(column + " duplicated to " + name)

            if len(df) > 1:
                st.markdown("--- \n ### Keep First *n* Rows only")
                keep_rows = st.slider("Number of rows to keep", 1, len(df))
                if st.button("Drop Remaining"):
                    len_old = len(df)
                    DFDAO().set(df.head(keep_rows), project)
                    st.info(f"Dropped {len_old - len(df)} rows")

            st.markdown("--- \n ### Drop rows by value")
            drop_column = st.selectbox("Select filter column", df.columns)
            drop_value = st.selectbox("Drop rows with value", list(df[drop_column].unique()))
            if st.button("Drop rows"):
                len_old = len(df)
                df = df[df[drop_column] != drop_value]
                DFDAO().set(df, project)
                st.info(f"Dropped {len_old - len(df)} rows")

            st.markdown("--- \n ### Move NaNs to Bottom")
            drop_column = st.selectbox("Select column", df.columns)
            if st.button("Move NaNs to Bottom"):
                df[drop_column] = df[drop_column].dropna().reset_index()
                DFDAO().set(df, project)

            st.markdown("--- \n ### Shuffle Rows")
            shuffle_columns = st.multiselect("Select columns to shuffle synchronously", list(df.columns))
            if st.button("Shuffle Rows"):
                df_temp = df[shuffle_columns].sample(frac=1)
                df_temp.reset_index(drop=True, inplace=True)
                df[shuffle_columns] = df_temp
                DFDAO().set(df, project)
                st.info("Shuffled rows of columns: " + ", ".join(shuffle_columns))

            st.markdown("--- \n ### Order By")
            ignore_cols = [
                "Class Names", "Class Indices", "Class Colors", "Models",
                "Training Configurations", "Overlap Percentage"
            ]
            order_columns = [col for col in df.columns if col not in ignore_cols]
            order_column = st.selectbox("Order table by column", order_columns)
            desc = st.checkbox("Descending", False)
            numeric = st.checkbox("Convert Column to numeric. Invalid values will result in NaN.")
            if numeric:
                max_rounding = st.number_input("Round to fixed number of decimal places", 0, 10, 4)
            if st.button("Order"):
                temp_df = df
                if numeric:
                    temp_df[order_column] = pd.to_numeric(temp_df[order_column], errors="coerce")
                    if max_rounding != 0:
                        temp_df[order_column] = temp_df[order_column].map(
                            ('{:,.' + str(max_rounding) + 'f}').format)
                temp_df[order_columns] = df[order_columns].sort_values(order_column,
                                                                       ascending=not desc).reset_index(drop=True)
                DFDAO().set(df, project)
                st.info("Ordered by " + order_column)

            st.markdown("--- \n ### Offset Columns")
            math_opts = {
                "Add": np.add,
                "Subtract": np.subtract,
                "Divide": np.divide,
                "Multiply": np.multiply
            }
            math_opt_selection = st.radio("Select Operation", list(math_opts.keys()))
            math_opt = math_opts[math_opt_selection]
            math_opt_in_1 = st.selectbox("Select Left Hand Side", list(df.columns))
            math_opt_in_2 = st.selectbox("Select Right Hand Side", list(df.columns))
            math_opt_out = st.text_input("Name of Output Column",
                                         f"{math_opt_in_1} {math_opt_selection} {math_opt_in_2}")
            if st.button("Math Op"):
                df[math_opt_out] = math_opt(df[math_opt_in_1], df[math_opt_in_2])
                DFDAO().set(df, project)
                st.info(f"Added column: {math_opt_out}")

        else:
            st.info("Editing not available. Project empty")


class SlimProjectWidget:
    def __init__(self):
        col1, col2 = st.columns([1, 2])
        current = Path(ProjectDAO().get()).stem
        projects = os_sorted(ProjectDAO().get_all())
        selection = col1.selectbox("Select Project", projects, projects.index(current) if current in projects else 0)
        if selection != current:
            ProjectDAO().set(selection)


class ImportExportTableWidget:
    def __init__(self):
        opt_col, import_col, export_col = st.columns(3)
        with opt_col:
            st.markdown("### ⭳⭱ Options")
            csv_args = {
                "sep": st.text_input(".CSV Column Separator", ";"),
                "decimal": st.text_input(".CSV Decimal Separator", ","),
                "header": 0 if st.checkbox(".CSV with Header", True) else None
            }
        with import_col:
            st.markdown("### Import table")
            name = st.text_input("Project Name. Warning! Setting the same name will overwrite.",
                                 Path(ProjectDAO().get()).stem)
            uploaded_file = st.file_uploader("Upload .csv file", type=".csv")
            if st.button("Import"):
                ProjectDAO().add(name, overwrite=True)
                DFDAO().set(pd.read_csv(uploaded_file, **csv_args), name)

        with export_col:
            st.markdown("### Export table")
            download_name = st.text_input("Export Name", Path(ProjectDAO().get()).stem)
            if st.button("Generate Download Link"):
                ExportWidget(download_name).df_link(csv_args)


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
        table = st.empty()
        max_items = min(len(df), 10)
        if len(df) > max_items:
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
        with st.spinner("Loading Table"):
            with table:
                AgGrid(
                    temp_df,
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
                st.write("Login")
                username = st.text_input("Username:", key="userlogin")
                password = st.text_input("Password:", type="password", key="userpw")
                st.form_submit_button("Login")

            result, username, password = LoginDAO().check_session(username, password)
            if password and not result:
                st.warning("Please enter valid credentials")

            if result:
                st.success(f"Logged in as: {username}")
                st.session_state.username = username
                st.session_state.password = password
                login_column.empty()
        return result


class BodyWidget:
    def __init__(self):
        try:
            project = ProjectDAO().get()
            df = DFDAO().get(project)

            with st.expander(f"🗀  {Path(ProjectDAO().get()).stem}"):

                SlimProjectWidget()

                TableWidget(df)

                st.write("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 🞥 Add Files")
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
                        DFDAO().set(df2, project, )

                with col2:
                    st.write("### ✎ Edit")
                    EditWidget(df)

                st.write("---")
                if st.checkbox("Show ⭳⭱ Options", False):
                    ImportExportTableWidget()

            if len(image_columns(df)):
                GalleryWidget(df)

            if df.empty:
                st.info("Start by uploading images to **🗀** the project, `\n` or use the **`Project`** module to "
                        "import .zip archives or locate files on the host system.")

        except:
            st.error("Display raised a message. "
                     "Try **'R'** to rerun")
            st.code(traceback.format_exc())


class ModuleWidget:
    def __init__(self):
        # Packages
        package_path = shelveutils.ModulePathDAO().get() or "pipelines"
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
        package_path = shelveutils.ModulePathDAO().get() or "pipelines"

        search = st.sidebar.text_input("What do you want to do?")
        with st.sidebar.expander("Settings"):
            module_path = shelveutils.ModulePathDAO().get()
            module_path = st.text_input("Module path", module_path or "pipelines")
            shelveutils.ModulePathDAO().set(module_path)

            log_path = shelveutils.LogPathDAO().get()
            log_path = st.text_input("Log path", log_path or "logs")
            shelveutils.LogPathDAO().set(log_path)

            data_path = shelveutils.DataPathDAO().get()
            # with st.form("Data path"):
            data_path = st.text_input("Data path", data_path or "data")
            #    if st.form_submit_button("Update data path"):
            shelveutils.DataPathDAO().set(data_path)

            st.write("---")

            if st.button("Reset Presets"):
                config.PresetListDAO().reset()
                config.ActivePresetDAO().reset()
                config.ConfigDAO().reset()
            st.write("---")

            upload_widget = st.empty()
            with upload_widget:
                uploader = FileUpload(str(package_path), f"Upload package", ".zip")

            if st.button(f"Upload package"):
                uploader.start()

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

            expander = st.sidebar.expander(pack_name, expanded=bool(search))

            modules = glob.glob(join(dirname(pack.__file__), "*.py"))
            modules = [basename(f)[:-3] for f in modules if isfile(f)
                       and not f.endswith('__init__.py')]

            for module_name in sorted(modules):
                if search and search.lower() not in module_name.lower():
                    continue

                full_module_name = f"{pack.__name__}.{module_name}"

                if expander.checkbox(module_name):
                    name = module_name.split('.')[-1]
                    shelveutils.BaseDAO.ACTIVE_PIPELINE = name
                    st.markdown(f"# {name}")
                    self.execute(full_module_name)

                yield full_module_name

    @staticmethod
    def execute(module_name):
        run_module(module_name)


