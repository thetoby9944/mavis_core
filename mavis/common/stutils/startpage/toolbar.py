import base64
import os
import shelve
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from mavis.stutils.widgets import FileUpload
from mavis import config
from mavis.pdutils import update
from mavis.shelveutils import current_project, project_path, create_new_project, delete_project, get_all_projects, \
    config_path
from mavis.stutils.workflow import Workflow


def toolbar():
    project = current_project()

    with shelve.open(project) as d:
        df = d["df"]

    edit_col, export_col, settings_col = st.beta_columns(3)
    with edit_col.beta_expander(" ✎ Edit"):
        if len(df):
            st.markdown("### Remove a column")
            columns_to_remove = st.multiselect("Remove column", list(df.columns))
            if st.button("Remove"):
                for column in columns_to_remove:
                    df = df.drop(column, axis=1)
                    st.info(column + " removed")
                update(df, project)

            st.markdown("--- \n ### Duplicate a column")
            column = st.selectbox("Duplicate column", df.columns)
            name = st.text_input("New Column Name")
            if st.button("Duplicate"):
                df[name] = df[column]
                update(df, project)
                st.info(column + " duplicated to " + name)

            st.markdown("--- \n ### Keep First *n* Rows only")
            keep_rows = st.slider("Number of rows to keep", 1, len(df))
            if st.button("Drop Remaining"):
                len_old = len(df)
                update(df.head(keep_rows), project)
                st.info(f"Dropped {len_old - len(d['df'])} rows")

            st.markdown("--- \n ### Drop rows by value")
            drop_column = st.selectbox("Select filter column", df.columns)
            drop_value = st.selectbox("Drop rows with value", list(df[drop_column].unique()))
            if st.button("Drop rows"):
                len_old = len(df)
                df = df[df[drop_column] != drop_value]
                update(df, project)
                st.info(f"Dropped {len_old - len(df)} rows")

            st.markdown("--- \n ### Shuffle Rows")
            shuffle_columns = st.multiselect("Select columns to shuffle synchronously", list(df.columns))
            if st.button("Shuffle Rows"):
                df_temp = df[shuffle_columns].sample(frac=1)
                df_temp.reset_index(drop=True, inplace=True)
                df[shuffle_columns] = df_temp
                update(df, project)
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
                        temp_df[order_column] = temp_df[order_column].map(('{:,.'+str(max_rounding)+'f}').format)
                temp_df[order_columns] = df[order_columns].sort_values(order_column,
                                                                       ascending=not desc).reset_index(drop=True)
                update(df, project)
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
            math_opt_out = st.text_input("Name of Output Column", f"{math_opt_in_1} {math_opt_selection} {math_opt_in_2}")
            if st.button("Math Op"):
                df[math_opt_out] = math_opt(df[math_opt_in_1], df[math_opt_in_2])
                update(df, project)
                st.info(f"Added column: {math_opt_out}")

        else:
            st.info("Editing not available. Project empty")

    with export_col.beta_expander("⭳⭱ .csv"):
        st.markdown("### Options")
        csv_args ={
            "sep": st.text_input(".CSV Column Separator", ";"),
            "decimal": st.text_input(".CSV Decimal Separator", ","),
            "header": 0 if st.checkbox(".CSV with Header", True) else None
        }
        st.markdown("--- \n ### Import table")
        uploaded_file = st.file_uploader("Upload .csv file", type=".csv")
        name = st.text_input("New Project Name", Path(project).stem + " (2)")
        if st.button("Import"):
            with shelve.open(project_path(name)) as new:
                new["df"] = pd.read_csv(uploaded_file, **csv_args)
            st.info("Project Created")

        st.markdown("--- \n ### Export table")

        if st.button("Generate Download Link"):
            csv_args["header"] = True
            csv = df.to_csv(index=False, **csv_args)
            b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
            href = f'<a download="{Path(project).stem}.csv" href="data:file/csv;base64,{b64}">' \
                   f'Download {Path(project).stem}.csv</a>'
            st.markdown(href, unsafe_allow_html=True)

    with settings_col.beta_expander("⚙️  Settings"):
        uploader = FileUpload("pipelines/Custom", "Upload Module", ".py")
        if st.button("Upload Modules"):
            uploader.start()

        st.markdown("### Presets")
        config.c.select("Preset")

        st.markdown("--- \n### Workflows")
        w = Workflow(st)
        w.run()

        st.markdown("--- \n ### Projects ")
        with shelve.open(config_path()) as c:
            current_project_span = st.empty()
            st.markdown("#### Select Project")
            projects = get_all_projects()
            selection = st.selectbox("Project", projects)
            if st.button("Select Project"):
                c["Last"] = selection
                st.info("Selected " + selection)
                st.warning("Please Refresh Table")
            if st.checkbox(f"Mark Project '{selection}' for deletion"):
                if st.button(f"Delete {selection}"):
                    delete_project(c, selection)

            st.markdown("#### Create New Project")
            name = st.text_input("Project Name")
            if st.button("Create"):
                create_new_project(c, name)

            current_project_span.markdown(f'Currently selected: {c["Last"] if "Last" in c else "None"}')

        st.markdown("--- \n ### Users")

        def create():
            if st.button("Create User"):
                with shelve.open(str(Path("data") / "login")) as login:
                    if not username or not password:
                        st.info("Cannot Create empty User")
                        return
                    if username in login["passwords"]:
                        st.info("User already exists")
                        return
                    passwords = login["passwords"]
                    passwords[username] = password
                    login["passwords"] = passwords
                    st.info("Created User " + username)
                    os.mkdir(str(Path("data") / username))

        def delete():
            if st.button("Delete User"):
                with shelve.open(str(Path("data") / "login")) as login:
                    passwords = login["passwords"]
                    if username in passwords and passwords[username] == password:
                        del passwords[username]
                        login["passwords"] = passwords
                        st.info("Deleted User " + username)
                    else:
                        st.info("Wrong credentials for " + username)

        username = st.text_input("Username", "")
        password = st.text_input("Password:", "", type="password")
        create()

        delete()
