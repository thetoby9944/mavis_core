import sys
from pathlib import Path

import streamlit as st
import shutil

import mavis.db as db


class AppStoreWidget:
    def __init__(self, modules_by_id):
        from mavis.ui.widgets.workspace.file_handling import FileUpload

        package_path = db.ModulePathDAO().get()

        query_params = st.experimental_get_query_params()
        if len(query_params):
            package, module, code = [query_params[key][0] for key in ["package", "module", "code"]]

            target = Path(package_path) / package
            target.mkdir(parents=True, exist_ok=True)
            with open(target / f"{module}.py", "w") as f:
                import zlib
                import base64
                f.write(zlib.decompress(base64.b64decode(code)).decode())

            st.success(f"Installed {package} / {module}. Go to workspace.")
            st.experimental_set_query_params(**{})

        st.write("# MAVIS AppStore")
        st.write("## Currently Installed App Modules")
        st.write(modules_by_id)
        st.write("🞥 Add functionality")
        upload_widget = st.empty()
        with upload_widget:
            uploader = FileUpload(
                str(package_path), f"Upload package", ".zip",
                help="Add functionality by uploading zipped python packages. "
                # "The python packages can contain arbitrary python scripts."
                # "After uploading a package, all its contents will be added to the menu."
                # "Clicking on a script in the menu will import and execute that script. "
                # "Hence, preferably you add stremalit scripts that run on import."
            )

        if st.button(f"Upload package"):
            uploader.start()
            st.success("Press **`R`** to refresh")

        if not package_path.is_dir():
            return

        st.write("---")

        modules_zip_path = Path(db.config_path()).resolve() / "mavis_modules"
        shutil.make_archive(str(modules_zip_path), 'zip', package_path)
        archive_path = modules_zip_path.with_suffix(".zip")
        archive_name = archive_path.name
        with open(archive_path, "rb") as f:
            st.download_button(
                "⇩ Download modules",
                f, archive_name,
            )

        if sys.platform.startswith("win") and st.button("🔗 Share Modules"):
            import pythoncom
            import win32com.client as client

            pythoncom.CoInitialize()
            outlook = client.Dispatch("Outlook.Application")
            message = outlook.createItem(0)
            message.Attachments.Add(str(archive_path))
            message.Display()
