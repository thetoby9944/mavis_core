Why we use modules as plugins
=============================
Apps is a term for a piece of software that is commonly accessible via a platform.
Mavis has apps too, they are simply plain python modules.

Philosophy
-------------

MAVIS is written developer first. It's a place for your code.
You can look at the examples to find inspiration or solve a use case, but ultimately,
mavis is a framework and platform for your own scripts.

Modules are plain python files.
Mavis will create a menu from modules.
To save you from gray hair, extending mavis is simple as adding a python file.

Benefits
-------------


The easiest way to extend mavis is to open the plugin directory in your favorite IDE.
This opens all the possibilities which are limited e.g. in notebooks.
Modern programming is based on

- versioning
- linting
- refactoring
- code navigation

It's time to bring it back to data science.
And it's a one-liner

.. code-block:: bash

    > code ~/mavis/mavis_modules


