import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu

from mavis.ui.widgets.settings.activity import LogWidget
from mavis.ui.widgets.settings.path_config import PathConfigWidget
from mavis.ui.widgets.settings.projects import ProjectWidget
from mavis.ui.widgets.settings.users import UserWidget


class SettingsWidget:
    def __init__(self):
        path_config, activity, users, projects = all_strs = [
            "Settings",
            "Activity Logs",
            "Users",
            "Projects"
        ]
        selected = option_menu(
            None, all_strs,
            icons=["gear", 'table', "people", "list-task"],
            menu_icon=None,
            default_index=0,
            orientation="horizontal"
        )
        if selected == path_config:
            PathConfigWidget()


        if selected == activity:
            LogWidget()

        if selected == users:
            UserWidget()

        if selected == projects:
            ProjectWidget()
