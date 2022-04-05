import pandas as pd
import streamlit as st
from mavis.db import LogDAO
from st_aggrid import AgGrid, DataReturnMode, GridOptionsBuilder, GridUpdateMode


class LogWidget:
    def __init__(self):
        self.logs = LogDAO().get_all()[::-1]

        if not self.logs:
            st.info("No activity so far.")
            return

        log_range = st.number_input("Show logs", 1, 100000, 10)
        df = pd.DataFrame(self.logs).head(log_range).drop("Preset Values", axis=1)
        df["Index"] = range(min(len(df), log_range))
        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column()
        gb.configure_selection()
        log = AgGrid(
            df,
            width="100 %",
            fit_columns_on_grid_load = True,
            gridOptions=gb.build(),
            update_mode=GridUpdateMode.SELECTION_CHANGED,
            allow_unsafe_jscode=True,
            data_return_mode=DataReturnMode.AS_INPUT
        )

        for row in log["selected_rows"]:
            for key, value in self.logs[row["Index"]].items():
                st.info(key)
                st.write(value)


