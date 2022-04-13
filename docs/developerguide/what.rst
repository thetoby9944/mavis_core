Adding your code
================

.. admonition:: TL;DR

    Plain python modules are something everyone can work with. To add one to mavis install
    vs-code and run:

    .. code-block:: bash

        > code ~/mavis/mavis_modules/My_custom_package/My_new_module.py`

Minimal Example
-------------------


This guide contains minimal instructions to add your code snippets, notebooks, 
gists and utils to mavis. 
First, run Mavis

.. code-block:: bash

    conda activate mavis
    mavis

Put this code in a file called :code:`~/mavis/mavis_modules/Custom/My_first_module.py`
E.g. open

.. code-block:: bash

    code ~/mavis/mavis_modules/Custom/My_first_module.py

and paste

.. code-block:: python

    import pandas as pd
    from mavis.db import get_df, set_df

    df: pd.DataFrame = get_df()     # read the current project data
    ...                             # do some processing of the dataframe
    set_df(df)                      # Update the current project data


.. hint::

    It can alternatively contain any other python code that will be executed.
    Generally it's useful to work with the `mavis.db` DataFrame as this will be persisted
    and managed by mavis for you. That way you can use the file handling and project structure
    which is provided by mavis out-of-the-box.


You can now visit mavis on `localhost:8501 <localhost:8501>`_
In the workspace tab you will see a new navigation item called *Custom*.
It will contain a sub menu item called *My first module*.


This code will be run everytime you press **R**
or load the module from the menu.



Streamlit Example
------------------------

Mavis is built on `streamlit <https://www.streamlit.io>`_.
If you want to include UI elements in your app, it's a simple as:

.. code-block:: python

    import pandas as pd
    import streamlit as st
    from mavis.db import get_df, set_df


    df: pd.DataFrame = get_df()         # read the current project data
    ...                                 #  do some processing of the dataframe
    if st.button("Update Project"):     # Only perform this action when the button is clicked
        set_df(df)                      # Update the current project data

The next time you will run this module, there will be a button. 
The dataframe will only be updated when the button is pressed. 
