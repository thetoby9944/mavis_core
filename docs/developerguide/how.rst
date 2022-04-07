
How do I add modules?
=========================

Remember: modules (apps, plugins, extensions) are plain :code:`.py` files.

Plugin architecture
---------------------------

The plugin structure looks as follows

.. code-block::

    .                     <- plugin root (you can configure this in mavis settings)
    |- Package_name       <- This name will appear in the main menu
    |  |
    |  |- My_module.py    <- This name will appear in the sub-menu as [My module]
    |  `- Other_module.py
    |
    `- Other_package
       |
        ...

By default, the plugin directory is located in

.. code-block::

    cd ~/mavis/mavis_modules

If you have :code:`vs-code` installed, getting started
with module development is as simple running

.. code-block::

    code ~/mavis/mavis_modules




Developing on a remote server
----------------------------

If you are running on a remote host there are multiple
options to bring your modules to mavis.

- You can upload any python file via mavis UI.
- The second, more robust approach is to develop locally and version the code e.g. via git or mercurial.
From the remote host you can pull any changes once you have sufficiently tested
your new module. The app will hot-reload automatically.


