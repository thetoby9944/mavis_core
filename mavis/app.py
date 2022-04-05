import base64
import os
import re
from pathlib import Path

import streamlit as st


def init_streamlit():
    from pathlib import Path
    import sys

    prefer_local_install = Path("..").resolve(strict=True).__str__()
    if prefer_local_install not in sys.path:
        sys.path.insert(0, prefer_local_install)
        print(sys.path)

    st.set_page_config(
        page_title="MAVIS",
        page_icon="assets/images/M_icon.png",
        layout="wide"
    )

    def get_base64(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()

    background_image_style = '''
        <style>
            .stApp {
                content: "";
                background-image: url("data:image/png;base64,''' + get_base64(Path(
        "assets/images/bg_img2.jpg")) + '''");
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-position: right top;
            }
        </style>
    '''

    nav_bar_style = """
        <style>
            div[data-stale="false"] > iframe[title="hydralit_components.NavBar.nav_bar"] {
            border-radius: 10px;
        }
        </style>
    """

    decoration_bar_style = '''
        <style>
            .css-kywgdc {
                position: absolute;
                top: 0px;
                right: 0px;
                left: 0px;
                z-index: 1001 !important;  
                height: 0.125rem;
                background-image: linear-gradient(
            90deg
            , #04f5e4, #0080ec);
                z-index: 1000020;
            }
        </style>
    '''

    sidebar_stlye = '''
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
    '''

    image_rendering_style = """
    <Style>
        img {image-rendering: pixelated;}
    </Style>
    """

    menu_gradient_style = """
    <Style>
        :root{
            --primary-color: blue;
        }
    </Style>
    """

    for style in [
        # decoration_bar_style,
        nav_bar_style,
        sidebar_stlye,
        menu_gradient_style,
        image_rendering_style,
    ]:
        st.markdown(style, unsafe_allow_html=True)

    with st.sidebar:
        st.write("# ")
        st.image("assets\\images\\MAVIS_logo.png", width=200)
        st.session_state["settings_placeholder"] = st.empty().container()
        st.session_state["settings_form_placeholder"] = st.empty().form("FORM")

        st.write("***")
        st.write("## Machine Learning & Computer Vision")
        st.write("*Rapid Prototyping and Technology Transfer from Research to Industries*  \n"
                 "Avoiding duplicate code **`est. 2020`**")


def process_exists(process_name):
    '''
    Check if process currently exists in OS System Takslist
    '''
    import subprocess
    import platform
    # tf.compat.v1.enable_eager_execution()

    current_platform = platform.system()
    if current_platform == "Windows":
        call = 'TASKLIST /FI "IMAGENAME eq ' + process_name + '"'
        run_obj = subprocess.run(call, capture_output=True)
        if re.search(process_name,
                     run_obj.stdout.decode('utf-8', 'backslashreplace')):
            return True
        else:
            return False
    else:
        p = subprocess.Popen(['ps', '-A'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        out = out.decode('utf-8', 'backslashreplace')
        for line in out.splitlines():
            if process_name in line:
                return True
        return False


@st.cache()
def init_tensorflow():
    """
        Tensorflow configuration,

        initializes devices, without blocking to reserve all availbale memory
        To disable GPU add:


            import os
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    """
    os.environ['TF_KERAS'] = '1'

    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                pass
                # tf.config.experimental.set_virtual_device_configuration(
                #    gpu,
                #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)]
                # )
        except RuntimeError as e:
            print(e)

    from db import LogPathDAO
    logs_base_dir = str(LogPathDAO().get().resolve())

    if process_exists('tensorboard.exe'):
        pass
    elif process_exists('tensorboard'):
        pass
    else:
        print("launch tensorboard process...")
        import subprocess
        subprocess.Popen(
            args=["tensorboard", "--logdir", logs_base_dir, "--bind_all", "serve"]
        )


@st.cache()
def init_labelstudio():
    if process_exists('label-studio.exe'):
        pass
    if process_exists('label-studio'):
        pass
    else:
        print("launch label-studio process...")
        import subprocess
        subprocess.Popen(
            args=["label-studio"]
        )


def run():
    """
    Run function in case you want to start mavis from within a python function
    Returns
    -------

    """
    dirname = Path(__file__).parent
    filename = 'app.py'

    # _config.set_option("server.headless", True)
    args = [str(Path(os.curdir).resolve()), str(Path(__file__).resolve())]
    os.chdir(dirname)
    print("Running streamlit from", os.getcwd())

    os.system(f"streamlit run {filename}")


if __name__ == "__main__":
    import streamlit as st

    init_streamlit()
    init_tensorflow()
    init_labelstudio()

    from mavis.ui.widgets.main import ModuleWidget
    from mavis.ui.widgets.login import LoginWidget

    if LoginWidget().check():
        ModuleWidget().execute()
