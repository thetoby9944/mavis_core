import streamlit as st
import shelve
from pathlib import Path

from mavis.stutils.sessionstate import get


class Login:
    def __init__(self):
        self.session_state = get(username="default", password='')

        with shelve.open(str(Path("data") / "login")) as login:
            if "passwords" not in login:
                login["passwords"] = {"default": "wg"}

    def check_session(self):
        with shelve.open(str(Path("data") / "login")) as login:
            username, password = self.session_state.username, self.session_state.password
            check_user = username in login["passwords"] and password == login["passwords"][username]
            return check_user, username, password

    def check(self):
        check_user, username, password = self.check_session()
        if not check_user:
            with st.beta_columns(3)[1]:
                with st.beta_expander("Login", True):
                    username_field = st.empty()
                    password_field = st.empty()
                    st.button("Login")
                    username = username_field.text_input("Username:", username, key="userlogin")
                    password = password_field.text_input("Password:", password, type="password", key="userpw")
                    self.session_state.username = username
                    self.session_state.password = password
                    if password and not self.check_session()[0]:
                        st.warning("Please enter valid credentials")

        return self.check_session()[0]


