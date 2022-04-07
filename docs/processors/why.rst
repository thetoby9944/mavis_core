Why subclass a mavis Processor?
================================

Using mavis to manage scripts is okay-ish but this isn't even its final form.
The true power of mavis unfolds after writing plugins that base on mavis' own processors.

It is very likely that you can remove 50% of your scripts code,
just by inherting from the built in processors.

We all hate writing boiler-plate code.
That's why mavis takes care of the most common data science best practices for you.

- Automatically follow ML Ops
- Keep a consistent look and feel for mavis
- Let mavis do the heavy lifting of data science

All this is achieved by subclassing a mavis processor.
Aditionally, why shouldn't you be using them?
They are as easy to use as:

.. code-block:: python

    > pip install mavis
    > python
    >>> from mavis.ui.processors import ImageProcessor


