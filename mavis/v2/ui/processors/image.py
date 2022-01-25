import glob
import json
import os
import types
from abc import ABC
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List

import PIL
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from pdutils import fill_column, overwrite_modes
from db import save_pil, LogDAO
from ui.processors.base import BaseProcessor


class ImageProcessor(BaseProcessor, ABC):
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
            save_numeric=False,

            multiprocessing=True,
    ):
        super().__init__(
            input_labels=input_labels,
            inputs_column_filter=inputs_column_filter,
            output_label=output_label,
            output_new_dir=output_new_dir,
            output_suffix=output_suffix,
            class_info_required=class_info_required,
            class_subset_required=class_subset_required,
            color_info_required=color_info_required,
            preview=preview,
            save_numeric=save_numeric
        )
        self.multiprocessing = multiprocessing
        self.flatten_result = False

    def process_one(self, *args):
        """
        Gets for every n in len(self.df):

            *[self.df[c].dropna().iloc[n]
              for c in self.input_columns]

        as arguments

        Should Return a PIL Image
        """
        raise NotImplementedError

    def preview(self, n):
        input_paths = list(zip(*self.input_args()))
        # Inputs
        try:
            with self.preview_columns[0]:
                st.image(Image.open(input_paths[n][0]), use_column_width=True)
            st.write("Inputs:")
            st.code("\n".join(input_paths[n]))
        except:
            pass

        # Outputs
        res = self.process_one(*input_paths[n])
        with self.preview_columns[1]:
            if not self.save_numeric:
                image_placeholder = st.empty()
                if isinstance(res, types.GeneratorType):
                    image_placeholder.image(next(res), use_column_width=True)
                elif isinstance(res, Image.Image):
                    image_placeholder.image(res, use_column_width=True)
            elif self.save_numeric:
                st.write(res)

    def tasks(self, args):
        val = self.process_one(*args)
        if not self.save_numeric and not self.inplace:
            val = save_pil(val, args[0], self.new_dir, self.suffix, self.column_out)
        return val

    def process_all(self):

        def prog_perc(x):
            return x / (len(inputs_paths) - 1) if len(inputs_paths) > 1 else 1.

        self.inplace = self.column_out is None
        if not self.inplace:
            self.df[self.column_out] = np.nan

        inputs_paths = list(zip(*self.input_args()))

        with ThreadPool(processes=max(os.cpu_count() - 1, 1) if self.multiprocessing else 1) as pool:
            st.info("Scheduling Tasks")
            bar = st.progress(0)
            results = []
            for i, args in enumerate(inputs_paths):

                results.append(pool.apply_async(self.tasks, (args,)))
                bar.progress(prog_perc(i))

            bar.progress(1.)

            st.info("Retrieving Results")
            bar = st.progress(0)
            for i, ans in enumerate(results):
                val = ans.get()
                if not self.inplace:

                    if type(val) is list:
                        self.df = fill_column(self.df, self.column_out, val, overwrite_modes["opt_a"])
                    elif type(val) is dict:
                        self.df.loc[i, self.column_out] = json.dumps(val)
                        self.flatten_result = True
                    else:
                        self.df.loc[i, self.column_out] = val

                bar.progress(prog_perc(i))
            bar.progress(1.)

        if self.flatten_result:
            temp_df = pd.json_normalize(self.df[self.column_out].dropna().apply(json.loads))
            temp_df.index = self.df[self.column_out].dropna().index
            self.df[temp_df.columns] = temp_df
            self.df = self.df.drop(self.column_out, axis=1)

        return self.df

    def run(self):
        self.preview_block()
        st.write("--- \n ### Run ")
        st.markdown("Start Processing")
        if st.button(" ▶ "):
            LogDAO(self.input_columns, self.column_out).add("Batch Run")
            df2 = self.process_all()
            self.save_new_df(df2)


class ImageGroupProcessor(ImageProcessor, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_labels=["Select Input Paths"],
            preview=False,
            multiprocessing=False,
            *args, **kwargs
        )

        self.split_at = st.text_input("Split path names at", "_")
        sample_path = self.input_args()[0][0]
        sample_path_parts = self.parts(sample_path)

        self.path_parts_indices_with_samples = list(zip(
            range(len(sample_path_parts)),
            sample_path_parts)
        )

        group_values = st.multiselect(
            "Grouping Identifier for images at positions",
            self.path_parts_indices_with_samples,
            self.path_parts_indices_with_samples
        )
        self.group_positions = [
            i
            for i, val in group_values
            if (i, val) in self.path_parts_indices_with_samples
        ]
        self.processed = {}

    def parts(self, full_path):
        return Path(full_path).stem.split(self.split_at)

    def tasks(self, args):
        # Local Vars
        path = args[0]

        # Get grouped image name and check if done
        base_image_name = self.split_at.join([self.parts(path)[int(i)] for i in self.group_positions])
        if str(Path(path).parent) + base_image_name in self.processed:
            return None

        group_selector = f"*{base_image_name.replace(self.split_at, '*')}*"
        paths = glob.glob(str(Path(path).parent / group_selector))
        result = self.process_group(paths)

        val = save_pil(
            img=result,
            base_path=path,
            new_dir=True,
            suffix=self.suffix,
            dir_name=self.column_out,
            stem=base_image_name
        )

        self.processed[str(Path(path).parent) + base_image_name] = True
        return val

    def process_group(self, paths: List[str]) -> PIL.Image:
        raise NotImplementedError

    def process_one(self):
        """Process one is replaced by process_group in the GroupImageProcessor"""
        pass