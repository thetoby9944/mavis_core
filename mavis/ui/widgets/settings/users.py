import streamlit as st

from mavis.db import UserDAO


class UserWidget:
    def __init__(self):
        st.markdown("### Users")

        username = st.text_input("Username", "")
        password = st.text_input("Password:", "", type="password")
        if st.button("Create User"):
            UserDAO().create(username, password)
        if st.button("Delete User"):
            UserDAO().delete(username, password)

