from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode


class OptionGrid:
    def __init__(self, options: List[Path], label="Option", default=""):
        self.label = label
        self.options = options
        self.default = default

        options_df = pd.DataFrame([{label: value} for value in options])
        gb = GridOptionsBuilder.from_dataframe(options_df)
        gb.configure_selection(suppressRowDeselection=True)

        gb.configure_default_column(
            min_column_width=300,
            width=300,
            defaultWidth=300,
            maxWidth=350,
            groupable=True,
            resizeable=True,
            dndSource=False,
            editable=True,
            cellStyle={"direction": "rtl", "text-overflow": "ellipsis"}
        )
        self.grid = AgGrid(
            options_df,
            gridOptions=gb.build(),
            fit_columns_on_grid_load=False,
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            data_return_mode=DataReturnMode.AS_INPUT,
            height=200
        )

    def selection(self):
        selected_rows = self.grid["selected_rows"]

        if len(selected_rows):
            selection = selected_rows[0].get(self.label, "")
        else:
            selection = self.default
        selection = Path(selection)
        st.write("Selected")
        st.code(selection.name)
        return selection


class ListView:
    def __init__(self, options: List[str], default: List[str], label):
        self.label = label
        self.options = options

        new_defaults = st.session_state.get(f"{label}_defaults_ListView", "") != default
        if new_defaults:
            self.default = st.session_state[f"{label}_defaults_ListView"] = default
        else:
            self.default = st.session_state.get(f"{label}_current_ListView", None)
            if self.default is None:
                self.default = default

    def _grid_layout(self, n_rows, column_spec):
        grid_layout = [
            [
                column.empty()
                for column in st.columns(column_spec)
            ]
            for _ in range(n_rows)
        ]
        return grid_layout

    def _option_grid(self, default_options):
        new_list = default_options.copy()
        for i, option in enumerate(default_options):
            option_col, move_down_col, move_up_col, del_col = st.columns((3,1,1,1))
            with option_col:
                st.code(f"{i+1}. {option}")
            with move_down_col:
                if i < (len(default_options) - 1) and st.form_submit_button(f"▼ {i+1}. "):#, key=f"{option}{i}down"):
                    new_list[i + 1], new_list[i] = default_options[i], default_options[i + 1]
            with move_up_col:
                if i > 0 and st.form_submit_button(f"▲ {i+1}. "):#, key=f"{option}{i}up"):
                    new_list[i], new_list[i - 1] = default_options[i - 1], default_options[i]
            with del_col:
                if st.form_submit_button(f"🞭 {i+1}. "):#, key=f"{option}{i}delete"):
                    del new_list[i]
        return new_list

    def selection(self):
        new_options = st.multiselect(
            self.label,
            self.options,
        )

        if st.form_submit_button(f"Update {self.label}"):
            self.default.extend(new_options)

        new_list = self._option_grid(
            default_options=self.default
        )
        st.session_state[f"{self.label}_current_ListView"] = new_list

        if new_list != self.default:
            st.experimental_rerun()

        return new_list


def icon(icon_name):
    st.markdown(f'<span class="material-icons" style="color:blue;">{icon_name}</span>', unsafe_allow_html=True)


def rgb_picker(label, value):
    hex_str = st.color_picker(label, value)
    return tuple([int(hex_str.lstrip("#")[i:i + 2], 16)
                  for i in (0, 2, 4)])


def identifier_options(sample_path, default_ops=None):
    parts = Path(sample_path).parts
    return [
        parts.index(sel)
        for sel in st.multiselect(
            "Include names of parent directories:",
            parts,
            default=[part for part in default_ops if part in parts]
        )
    ]


def contour_filter_options(min_circularity=0., min_area=0, max_area=0, ignore_border_cnt=True):
    min_circularity = st.number_input("Minimum circularity for detected segments", 0., 1., min_circularity)
    min_area = st.number_input("Minimum Area for detected segments. Set to zero to disable.", 0, 100000, min_area)
    max_area = st.number_input("Maximum Area for detected segments. Set to zero to disable.", 0, 100000, max_area)
    ignore_border_cnt = st.checkbox("Ignore Border Contours", ignore_border_cnt)
    return min_circularity, min_area, max_area, ignore_border_cnt
