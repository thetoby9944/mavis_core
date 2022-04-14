import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image
from natsort import os_sorted
from st_aggrid import GridOptionsBuilder, AgGrid, JsCode, GridUpdateMode
from streamlit_option_menu import option_menu

from mavis.db import ProjectDAO, DFDAO
from mavis.pdutils import image_columns
from mavis.pilutils import IMAGE_FILETYPE_EXTENSIONS
from mavis.ui.widgets.workspace.file_handling import ExportWidget


class HomeWidget:
    def __init__(self):
        try:
            heading = st.empty()

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

            project = ProjectDAO().get()
            heading.write(f"# 🗀  {Path(ProjectDAO().get()).stem}")


            with col1:

                selected = option_menu(
                    None, ["Add Files", "Project"],
                    icons=["plus", 'table'],
                    menu_icon="home",
                    default_index=0,
                    orientation="horizontal"
                )
            if selected == "Project":
                df = DFDAO().get(project)
                TableWidget(df)

                if df.empty:
                    cols = st.columns(3)
                    with cols[1]:
                        st.image("assets/images/wowsuchemtpy.png")
                        st.write("### Your project is empty")
                        st.info(
                            "Start by adding files to **🗀** the project. `\n` "
                            "You can upload files, import and extract .zip archives, "
                            "and locate files on the host system by folder or RegEx.  \n"
                            "In case you are running on an Windows localhost, "
                            "you can also use your native folder browser."
                            " Click **`[ADD FILES]`** above to conintue"
                        )

            if selected == "Add Files":
                st.write("***")
                from mavis.ui.widgets.workspace.file_handling import FileImportWidget
                FileImportWidget()

        except:
            st.error(
                "Display raised a message. "
                "Try **'R'** to rerun"
            )
            st.button("Refresh")
            with st.expander("Message"):
                st.code(traceback.format_exc())


class SlimProjectWidget:
    def __init__(self):
        current = Path(ProjectDAO().get()).stem
        projects = os_sorted(ProjectDAO().get_all())
        selection = st.selectbox("Select Project", projects, projects.index(current) if current in projects else 0)
        if selection != current:
            ProjectDAO().set(selection)


class TableWidget:
    def __init__(self, df):
        if len(df) == 0:
            return

        max_items = min(len(df), 10)
        omit_col, import_col, export_col = st.columns([4, 6, 2])
        with omit_col:
            if len(df) > max_items:
                max_items = int(st.number_input(
                    "Maximum number of rows to display",
                    0, value=max_items,
                    help="Set to 0 to show all rows"
                ))
                max_items = max_items if max_items else len(df)

        csv_args = {
            "sep": ";",
            "decimal": ",",
            "header": 1
        }
        name = Path(ProjectDAO().get()).stem

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
        grid.configure_selection("single")
        for img_column in image_columns(df):
            cell_renderer = JsCode(
                """function (params) {
                        var element = document.createElement("span");
                        var imageElement = document.createElement("img");
    
                        if (params.data."""+img_column.replace(" ", "_")+""") {
                            imageElement.src = params.data."""+img_column.replace(" ", "_")+r""";
                            imageElement.width="20";
                        } else {
                            imageElement.src = "D:\Schiele\MW\Local_Data\local_2021354_PVA_Intel_Marketing\2021354_FinePitch AutoTray Bildmaterial\FinePitch AutoTray Bildmaterial\ND5 VarM\t1-x1-p1-run1 channel1 TrayScan1 D85892CJ00002 FIRGate1_65,00MHz_70,00MHz-Scaled-Registered-Variation_Model.png";
                        }
                        element.appendChild(imageElement);
                        element.appendChild(document.createTextNode(params.value));
                        return element;
                    }
                """
            )
            # print(cell_renderer.js_code)
            # grid.configure_column(img_column, cellRenderer=cell_renderer)

        grid_response = AgGrid(
            temp_df,
            # update_mode=GridUpdateMode.MANUAL,
            gridOptions=grid.build(),
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
            height=330,
            update_mode=GridUpdateMode.SELECTION_CHANGED

        )

        omit_col, import_col, export_col = st.columns([4, 7, 1])
        with export_col:
            if st.button("Download .csv"):
                ExportWidget(f"{name}.csv").df_link(csv_args)

        if st.checkbox("Show Statistics"):
            st.write(df.describe())
        if st.checkbox("Show Info"):
            st.write(df.info())

        selection = grid_response['selected_rows']
        if selection:
            st.write("---")

        columns = st.columns(len(df.columns))
        for selected_row in selection:
            for st_column, (df_column, value) in zip(columns, selected_row.items()):
                with st_column:
                    st.write(f"**{df_column}**")

        columns = st.columns(len(df.columns))
        for selected_row in selection:
            for st_column, (df_column, value) in zip(columns, selected_row.items()):
                with st_column:
                    path_value = Path(value)
                    if path_value.is_file() and path_value.suffix in IMAGE_FILETYPE_EXTENSIONS:
                        st.image(Image.open(path_value))
                    else:
                        st.code(value)

        if selection:
            st.write("---")
            st.write(selection[0])
