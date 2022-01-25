from time import sleep

import streamlit as st


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
