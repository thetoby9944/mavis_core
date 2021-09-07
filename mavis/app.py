import base64
import os
from pathlib import Path

import tensorflow as tf


def init_streamlit():
    import streamlit as st
    from mavis import __version__

    st.set_page_config(
        page_title="Mavis",
        page_icon="assets/images/icon.png",
        layout="wide",
    )
    st.sidebar.image("assets/images/logo.svg", output_format="JPG", width=300)
    st.sidebar.code(f"\t\t\t  {__version__}")

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

    hide_decoration_bar_style = '''
        <style>
                header {visibility: hidden;}
        </style>
    '''

    st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)


def init_tensorflow():
    """
        Tensorflow configuration,

        initializes the devices.
        To disable GPU add:


            import os
            os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    """

    # tf.compat.v1.enable_eager_execution()

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


def run():
    dirname = Path(__file__).parent
    filename = 'app.py'

    # _config.set_option("server.headless", True)
    args = [str(Path(os.curdir).resolve()), str(Path(__file__).resolve())]
    os.chdir(dirname)
    print(os.getcwd())
    import streamlit.bootstrap

    # streamlit.cli.main_run(filename, args)
    streamlit.bootstrap.run(filename, '', args, flag_options={})


if __name__ == "__main__":
    init_streamlit()
    init_tensorflow()

    from stutils.widgets import LoginWidget, BodyWidget, ModuleWidget

    if LoginWidget().check():
        BodyWidget()
        list(ModuleWidget().get_modules_interactive())
