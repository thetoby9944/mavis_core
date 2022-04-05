import glob
import pkgutil
import sys
from collections import defaultdict
from os.path import join, dirname, basename, isfile
from runpy import run_module

import hydralit_components as hc

import streamlit as st
from streamlit_option_menu import option_menu

import mavis.db as db
from mavis.db import ModulePathDAO
from mavis.ui.widgets.app_store import AppStoreWidget
from mavis.ui.widgets.workspace.home import HomeWidget
from mavis.ui.widgets.settings import SettingsWidget
from mavis.ui.widgets.workflow import WorkflowWidget


class ModuleWidget:
    def __init__(self):
        # Packages
        package_path = ModulePathDAO().get() or "pipelines"
        # print(f"Looking for mavis modules in {package_path}")
        if str(package_path) not in sys.path:
            sys.path.append(str(package_path))

        self.packages = [
            __import__(modname, fromlist="dummy")
            for importer, modname, ispkg in pkgutil.iter_modules([str(package_path)], "")
        ]
        self.pack_names = [pack.__name__.split('.')[-1] for pack in self.packages]

    def get_modules(self):
        for pack, pack_name in zip(self.packages, self.pack_names):

            modules = glob.glob(join(dirname(pack.__file__), "*.py"))
            modules = [basename(f)[:-3] for f in modules if isfile(f)
                       and not f.endswith('__init__.py')]

            for module_name in sorted(modules):
                full_module_name = f"{pack.__name__}.{module_name}"
                yield full_module_name

    def get_modules_interactive(self):
        search = st.sidebar.text_input(f"What do you want to do?")
        for pack, pack_name in zip(self.packages, self.pack_names):

            modules = glob.glob(join(dirname(pack.__file__), "*.py"))
            modules = [basename(f)[:-3] for f in modules if isfile(f)
                       and not f.endswith('__init__.py')]

            for module_name in sorted(modules):
                if search and search.lower() not in module_name.lower():
                    continue

                full_module_name = f"{pack.__name__}.{module_name}"

                yield full_module_name, module_name, pack_name

    def execute(self):
        query_params = st.experimental_get_query_params()
        default_index = 2 if len(query_params) else 0

        workspace_str, workflow_design_str, apps_str, settings_str = menu_strs = [
            "Workspace", "Workflow Design", "Apps", 'Mavis'
        ]

        modules_by_pack = defaultdict(list)
        modules_by_id = {}

        for full_module_name, module_name, pack_name in self.get_modules_interactive():
            module_name = module_name.replace("_", " ")
            pack_name = pack_name.replace("_", " ")

            modules_by_pack[pack_name] += [(full_module_name, module_name, pack_name)]
            modules_by_id[full_module_name] = [module_name, pack_name]

        with st.sidebar:
            selected = option_menu(
                None, menu_strs,
                icons=['house', "stack", 'bag-fill', 'gear'],
                menu_icon="cast",
                default_index=default_index,
                orientation="vertical"
            )

        if selected == workspace_str:
            # st.sidebar.write("***")
            menu_data = [
                {
                    'label': pack_name,
                    'submenu': [
                        {
                            'label': module_name,
                            "id": full_module_name

                        }
                        for full_module_name, module_name, pack_name
                        in pack
                    ]

                }
                for pack_name, pack
                in modules_by_pack.items()
            ]

            # over_theme = {'txc_inactive': '#FFFFFF'}
            over_theme = {
                'txc_inactive': 'var(--text-color)',
                'menu_background': 'var(--secondary-background-color)', # '#FFFFFF',
                'txc_active': "var(--primary-color)", #"#FFFFFF",  # white '#283d5a',  # dark blue  #
                "option_active": 'var(--secondary-background-color)', # "linear-gradient(90deg, #020024 0%, #090979 35%, #00d4ff 100%);"

                # "#bd37ba" #"#ff0060" #"#ff00ae" magenta# "#66DDEE" teal # "#FF5757"  # red  "#f4f8ff"  # light gray  #
                # 'option_active':'blue'
            }

            menu_id = hc.nav_bar(
                menu_definition=menu_data,
                override_theme=over_theme,
                sticky_nav=True,
                sticky_mode="pinned",
                home_name="Home",
                hide_streamlit_markers=False,
                option_menu=True,
                use_animation=False,
            )

            if menu_id in modules_by_id:
                name, full_name = modules_by_id[menu_id]
                db.BaseDAO.ACTIVE_PIPELINE = name
                st.markdown(f"# {name}")
                run_module(menu_id)

            if menu_id == "Home":
                HomeWidget()

        if selected == workflow_design_str:
            WorkflowWidget()

        if selected == apps_str:
            AppStoreWidget({
                module: [pack_element[1] for pack_element in pack]
                for module, pack in modules_by_pack.items()
            })

        if selected == settings_str:
            SettingsWidget()
