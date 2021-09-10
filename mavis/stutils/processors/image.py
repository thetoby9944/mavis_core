import json
import os
import types
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from pdutils import fill_column, overwrite_modes
from shelveutils import save_pil, LogDAO
from stutils.processors.base import BaseProcessor


class ImageProcessor(BaseProcessor):
    def __init__(self, multiprocessing=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
            LogDAO(self.input_columns, self.column_out).add()
            df2 = self.process_all()
            self.save_new_df(df2)
