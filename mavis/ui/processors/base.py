import traceback

import streamlit as st

import config
from pdutils import image_columns
from db import ProjectDAO, DFDAO

preview_global = 0


class MultiProcessor:
    def __init__(self, name, inputs):
        inputs[st.selectbox(name, list(inputs.keys()))]()


class BaseProcessor:
    def __init__(
            self,
            input_labels=None,
            inputs_column_filter=None,
            output_label=None,
            output_new_dir=True,
            output_suffix="",
            class_info_required=False,
            class_subset_required=False,
            color_info_required=True,
            preview=True,
            save_numeric=False
    ):

        if self.__doc__:
            with st.expander("Description"):
                st.markdown(self.__doc__, unsafe_allow_html=True)

        self.save_numeric = save_numeric
        self.create_preview = preview
        self.preview_columns = []
        self.column_out = None
        self.input_columns = None
        self.suffix = ""
        self.new_dir = True
        self.inplace = False
        self.df = DFDAO.get(ProjectDAO().get())


        if input_labels is not None:
            st.markdown("### Data")
            if inputs_column_filter is None:
                inputs_column_filter = [image_columns] * len(input_labels)

            self.input_columns = [
                st.selectbox(label, column_filter(self.df) if column_filter is not None else self.df.columns)
                for label, column_filter in zip(input_labels, inputs_column_filter)
            ]

            if not all(self.input_columns):
                st.info("Upload files first")
                self.create_preview = False

        if output_label is not None:
            self.column_out_block(label=output_label, new_dir=output_new_dir, suffix=output_suffix)

        st.markdown("--- \n ### Options")
        config.Preset().select("Select Preset")
        if class_info_required or class_subset_required:
            config.Preset().class_info_block(with_color=color_info_required)

            if class_subset_required:
                self.class_names, self.class_colors = config.Preset().class_subset_block()


    def save_new_df(self, df):
        self.df = DFDAO().set(df, ProjectDAO().get())

    def check_preview(self):
        global preview_global
        n = len(self.input_args(dropna_jointly=False)[0])

        if n > 1:
            preview_global = st.slider("Preview Image Number", 0, n - 1, preview_global)
            preview_batch_size = 1
            if st.checkbox("Adv. Options", False, key="adv_prev_options"):
                preview_batch_size = st.number_input("Preview Batch Size", 1, n - preview_global)
            return preview_global, preview_batch_size

        elif n == 1:
            return 0, 1

        else:
            st.warning("Input columns are empty")
            return None, 1

    def column_out_block(self, label="", new_dir=True, suffix=""):
        if label:
            self.column_out = st.text_input("Remember Results in", f"{self.input_columns[0]} {label}")
        if suffix:
            self.suffix = suffix
        elif label:
            use_suffix = True  # st.checkbox("Save results with suffix")
            self.suffix = f"-{label.replace(' ', '_')}.png" if use_suffix else ".png"
        self.new_dir = new_dir

    def preview_block(self, expanded=True):
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

    def input_args(self, dropna_jointly=True):
        temp_df = self.df[self.input_columns]
        if dropna_jointly:
            temp_df = temp_df.dropna()

        temp_df.columns = [f"{i}_{c}" for i, c in enumerate(temp_df.columns)]

        if dropna_jointly:
            res = [temp_df[c].values.tolist() for c in temp_df.columns]
        else:
            res = [temp_df[c].dropna().values.tolist() for c in temp_df.columns]

        return res
