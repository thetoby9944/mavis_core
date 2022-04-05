import glob
import json
import os
import traceback
import types
from abc import ABC
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

import PIL
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from mavis.config import DefaultSettings
from mavis.pdutils import fill_column, overwrite_modes
from mavis.db import save_pil, LogDAO
from mavis.ui.processors.base import BaseProcessor


class ImageProcessor(BaseProcessor, ABC):
    """
    Base Class for image processing. Inherit this class to create a batch processor.
    You must implement

    `def process_one(self, *args):`

    Further, you need to specify the settings that inherits from `DefaultSettings`.
    I.e.

    ```
    class MySettings(DefaultSettings):
        _input_labels = ["Select inputs"]
        _output_label = "Output"

    class MyImageProcessor(ImageProcessor):
        config = MySettings()

        def process_one(input):
            ...
    ```

    """
    config: DefaultSettings = ...

    def __init__(
            self,
            preview=True,
            save_numeric=False,
            multiprocessing=True,
            new_dir=True,
            *args, **kwargs
    ):
        super().__init__(
            preview=preview,
            save_numeric=save_numeric,
            new_dir=new_dir
            *args, **kwargs
        )
        self.multiprocessing = multiprocessing
        self.flatten_result = False

    def process_one(self, *args):
        """
        The processor iterates through the dataframe.
        On each row, it calls `process_one`.
        The config specifies how many args `process_one` will get.

        Example
        config has _input_labels = ["Select input 1", "Select input 2"]
        then `process_one` will be called with `*args == (input_1: str, input_2: str)`
        `input_1` and `input_2` will be the row-wise values from the selected columns.

        assert len(args) == len(self.config._input_labels)

        Returns a PIL Image or if `save_numeric` is set, a dictionary of values
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
        with self.preview_columns[1]:
            with st.spinner("Generating Preview"):
                res = self.process_one(*input_paths[n])
                if not self.save_numeric:
                    image_placeholder = st.empty().container()
                    if isinstance(res, types.GeneratorType):
                        image_placeholder.image(list(res), use_column_width=True)
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

    def core(self):
        self.preview_block()
        st.write("--- \n ### Run ")
        st.markdown("Start Processing")
        if st.button(" ▶ "):
            LogDAO(self.input_columns, self.column_out).add("Batch Run")
            df2 = self.process_all()
            self.save_new_df(df2)


class NumericalProcessor(ImageProcessor, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(
            save_numeric=True,
            *args, **kwargs,
        )


class SingleImageProcessor(ImageProcessor, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(
            preview=True,
            *args, **kwargs
        )

    def single_task(self, n: int) -> Optional[Union[Dict, Image.Image]]:
        input_paths = list(zip(*self.input_args(dropna=False)))[n]
        # Inputs
        try:
            st.write("Inputs:")
            st.code("\n".join(input_paths))
        except:
            pass

        # Outputs
        res = self.process_one(*input_paths)
        if res is None:
            return None

        st.write("#### Task Results")
        if not self.save_numeric:
            image_placeholder = st.empty()
            if isinstance(res, types.GeneratorType):
                image_placeholder.image(next(res))
            elif isinstance(res, Image.Image):
                image_placeholder.image(res)
        elif self.save_numeric:
            st.write(res)

        return res

    def single_task_block(self) -> Tuple[Optional[pd.DataFrame], int]:
        st.write("### Process Image")
        try:
            n, _ = self.check_preview(max_batch_size=1)
            if n is not None:
                task_result = self.single_task(n)
                return task_result, n
        except:
            st.warning("Processor Message")
            st.code(traceback.format_exc())
            st.stop()

    def core(self):
        task_result, n = self.single_task_block()
        if task_result is not None and st.button(" Save 💾"):
            if not self.save_numeric and not self.inplace:
                task_result = save_pil(
                    task_result,
                    self.input_args()[n][0],
                    self.new_dir,
                    self.suffix,
                    self.column_out
                )

            self.df.loc[n, self.column_out] = task_result
            LogDAO(self.input_columns, self.column_out).add("Single Image Edit")
            self.save_new_df(self.df)


class ImageGroupSettings(DefaultSettings):
    _processed = {}
    _path_part_indices_with_samples: List[Tuple[int, str]] = []

    split_at: str = "_"
    group_values: List[Tuple[int, str]] = []

    def parameter_block(self):
        super().parameter_block()
        self.split_at = st.text_input("Split path names at", self.split_at)
        sample_path = self.sample_path()
        sample_path_parts = self.parts(sample_path)

        self._path_part_indices_with_samples = list(zip(
            range(len(sample_path_parts)),
            sample_path_parts)
        )

        self.group_values = st.multiselect(
            "Grouping Identifier for images at positions",
            self._path_part_indices_with_samples,
            self.group_values
        )


    @property
    def group_positions(self):
        return [
            i
            for i, val in self.group_values
            if (i, val) in self._path_part_indices_with_samples
        ]

    def parts(self, full_path):
        return Path(full_path).stem.split(self.split_at)


class ImageGroupProcessor(ImageProcessor, ABC):
    config: ImageGroupSettings = None
    _processed = {}

    def __init__(self, *args, **kwargs):
        super().__init__(
            preview=False,
            multiprocessing=False,
            *args, **kwargs
        )

    def tasks(self, args):
        c = self.config
        # Local Vars
        path = args[0]

        # Get grouped image name and check if done
        base_image_name = c.split_at.join([c.parts(path)[int(i)] for i in c.group_positions])
        if str(Path(path).parent) + base_image_name in self._processed:
            return None

        group_selector = f"*{base_image_name.replace(c.split_at, '*')}*"
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

        self._processed[str(Path(path).parent) + base_image_name] = True
        return val

    def process_group(self, paths: List[str]) -> PIL.Image:
        raise NotImplementedError

    def process_one(self):
        """Process one is replaced by process_group in the GroupImageProcessor"""
        pass