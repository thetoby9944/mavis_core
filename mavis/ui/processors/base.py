import os
import shutil
import sys
import traceback
from abc import ABC
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict

import streamlit as st
from streamlit_option_menu import option_menu

from mavis.config import DefaultSettings, MultiProcessorSettings
from mavis.db import ProjectDAO, DFDAO, ModulePathDAO
from mavis.presets.base import PresetHandler


class BaseProcessor:
    config: DefaultSettings = None

    def __init__(
            self,
            new_dir=True,
            preview=True,
            save_numeric=False,
            **kwargs
    ):
        if kwargs:
            st.warning(
                "Found unhandled kwargs. "
                "This warning is to show that the implementation might use deprecated arguments"
            )
            st.warning(kwargs)

        if self.__doc__:
            with st.expander("Description"):
                st.markdown(self.__doc__, unsafe_allow_html=True)
        self.df = DFDAO.get(ProjectDAO().get())
        self.save_numeric = save_numeric
        self.create_preview = preview
        self.preview_columns = []
        self.new_dir = new_dir

        # Set after running DefaultSettings.parameter_block()
        self.column_out = ""
        self.suffix = ""
        self.input_columns = []
        self.inplace = False
        self.run()

    @staticmethod
    def progress_percentage(i: int, n: int):
        return min(1., i + 1 / n) if n > 0 and i < n else 1.

    def save_new_df(self, df) -> None:
        self.df = DFDAO().set(df, ProjectDAO().get())
        st.success("Success. Updated Project.")

    def check_preview(self, max_batch_size=8) -> Tuple[Optional[int], int]:
        n = len(self.input_args(dropna_jointly=False)[0])

        if n > 1:
            preview_number = int(st.number_input("Image Number", 0, n - 1))
            preview_batch_size = 1
            if max_batch_size > 1 and st.checkbox("Adv. Options", False, key="adv_prev_options"):
                preview_batch_size = st.number_input("Batch Size", 1, min(max_batch_size, n - preview_number))
            return preview_number, preview_batch_size

        elif n == 1:
            return 0, 1

        else:
            st.warning("Input columns are empty")
            return None, 1

    def preview_block(self, expanded=False) -> None:
        if self.create_preview:
            with st.expander("Preview", expanded):
                try:
                    self.preview_columns = st.columns(2)
                    n, batch_size = self.check_preview()
                    if n is not None:
                        for i in range(batch_size):
                            self.preview(n + i)
                except:
                    st.warning("Preview message")
                    st.code(traceback.format_exc())  # .split("\n")[-2])

    def preview(self, n) -> None:
        raise NotImplementedError

    def input_args(self, dropna_jointly=True, dropna=True, include_output_column=False) -> List[List[Any]]:
        temp_df = self.df[
            self.input_columns +
            ([self.column_out] if include_output_column and self.column_out in self.df else [])
        ]

        if dropna and dropna_jointly:
            temp_df = temp_df.dropna()

        temp_df.columns = [f"{i}_{c}" for i, c in enumerate(temp_df.columns)]

        if dropna and dropna_jointly or not dropna:
            res = [temp_df[c].values.tolist() for c in temp_df.columns]
        else:
            res = [temp_df[c].dropna().values.tolist() for c in temp_df.columns]

        return res

    def module_link(self, filename, folder, module):
        with open(filename) as f:
            code_str = f.read()
            import zlib
            import base64
            from urllib.parse import quote_plus
            params = (
                f"package={quote_plus(folder)}&"
                f"module={quote_plus(module)}&"
                f"code={quote_plus(base64.b64encode(zlib.compress(code_str.encode())))}"
            )
            url = (
                f"http://localhost:8501?"
                f"{params}"
            )
            st.code(url)

    def fork_and_edit(self):
        folder, module = self.__module__.split(".")
        module_path = ModulePathDAO().get() / folder
        filename = module_path / module
        filename = f'{filename.with_suffix(".py").resolve()}'
        target_name = f"Fork of {Path(filename).name}"  # st.text_input(f"Open fork as", )
        is_win_os = sys.platform.startswith('win')

        st.write("---")
        h_stack = st.columns([17, 1, 1, 1])
        if h_stack[-1].button(
                "🔗",
                help="Get Module Link"
        ):
            self.module_link(filename, folder, module)

        if is_win_os and h_stack[-2].button(
                "📄",
                help="Open the module for editing"
        ):
            os.startfile(filename)

        if is_win_os and h_stack[-3].button(
                "➦",
                help="Fork"
        ):
            target = module_path / target_name
            shutil.copyfile(filename, target)
            os.startfile(target)

    def core(self):
        raise NotImplementedError

    def run(self):
        assert self.config is not None, "The Processor does not have a configuration"
        self.config.set_df(self.df)

        try:
            st.session_state["settings_placeholder"].empty()
            with st.session_state["settings_placeholder"]:
                PresetHandler.select()
            self.config.parameter_block()
            PresetHandler.update(self.config)
        finally:
            try:
                st.session_state["settings_form_placeholder"].__exit__(None, None, None)
            except:
                pass

        self.column_out = self.config.column_out
        self.input_columns = self.config.input_columns
        self.inplace = (
                self.column_out is None
                or self.column_out in self.input_columns
        )
        self.suffix = self.config.suffix

        self.core()
        self.fork_and_edit()


class MultiProcessor(MultiProcessorSettings, ABC):
    """
    Sample usage

        class ClusteringProcessor(MultiProcessor):
            name = "Clustering Methods"

            @property
            def inputs(self):
                return {
                    "K-Means with Affinity Propagation": MeanShiftProcessor,
                    "Channel-Wise Histogram Binning": HistogramBinningProcessor,
                    "Median Filter for Saturation and Value": MedianBlurProcessor,
                }

    """
    def __init__(self, **data: Any):
        super().__init__(**data)
        options = list(self.inputs.keys())
        selection = option_menu(
            self.name if self.show_name else "",
            options,
            default_index=self.last_selection
        )
        self.last_selection = options.index(selection)
        self.update()
        self.inputs[selection]()

    @property
    def inputs(self) -> Dict[str, type(BaseProcessor)]:
        raise NotImplementedError
