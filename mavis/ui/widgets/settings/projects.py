import streamlit as st
from datetime import datetime

from mavis.db import ProjectDAO


class ProjectWidget:
    def __init__(self):

        st.markdown("### Create New Project")
        name = st.text_input("Project Name", f"{datetime.now():%y%m%d}")
        if st.button("Create"):
            ProjectDAO().add(name)

        st.markdown("### Delete Projects ")
        selection = st.selectbox("Delete Project", ProjectDAO().get_all())
        if st.checkbox(f"Mark {selection} for deletion") and st.button(f"Delete {selection}"):
            ProjectDAO().delete(selection)

