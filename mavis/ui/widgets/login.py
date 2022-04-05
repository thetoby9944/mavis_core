from time import sleep

import streamlit as st

from mavis.db import LoginDAO


def get(**kwargs):
    res = []
    for key, val in kwargs.items():
        if key not in st.session_state:
            st.session_state[key] = val
            sleep(1)
            assert st.session_state[key] == val
        res += [st.session_state[key]]

    if len(res) == 0:
        return st.session_state
    if len(res) == 1:
        return res[0]
    if len(res) >= 1:
        return tuple(res)


class LoginWidget:
    def check(self):
        login_placeholder = st.empty()
        login_column = login_placeholder.container()
        username, password = get(username="default", password="")
        result, username, password = LoginDAO().check_session(username, password)
        if not result:
            column = login_column.columns(3)[1]

            form = column.form("Login")
            with form:
                st.write("### Login")
                username = st.text_input("Username:", key="userlogin")
                password = st.text_input("Password:", type="password", key="userpw")
                if st.form_submit_button("Login"):

                    result, username, password = LoginDAO().check_session(username, password)
                    if password and not result:
                        st.warning("Please enter valid credentials")

                    if result:
                        st.success(f"Logged in as: {username}")
                        st.session_state.username = username
                        st.session_state.password = password
                        login_placeholder.empty()
                        # st.experimental_rerun()
        return result

