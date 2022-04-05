import pandas as pd
import streamlit as st

import mavis.db as db


class PathConfigWidget:
    def __init__(self):
        st.write("# ⚙ Settings")
        db.ModulePathDAO().edit_widget()
        db.LogPathDAO().edit_widget()
        db.DataPathDAO().edit_widget()

        st.write("---")
        st.write("### Reset")
        if st.button(
                f"Reset Presets",
                help=f"Resets {db.BaseDAO.ACTIVE_PIPELINE}"
        ):
            db.PresetListDAO().reset()
            db.ActivePresetDAO().reset()
            db.ModelDAO().reset()
            db.ConfigDAO().reset()

        st.write("---")
        st.write("### Versions")

        from mavis import __version__
        import tensorflow

        versions = pd.DataFrame([
            {f"Version": f"{__version__}"},
            {f"Version": f"{tensorflow.__version__}"}
        ], index=["Mavis", "Tensorflow"])


        st.write(versions)
        st.write("***")
        st.write("### About")
        st.code(
            "CC TOBIAS SCHIELE   \n"
            "⚖ MIT"
        )
