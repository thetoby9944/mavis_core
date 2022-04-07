.. Imageflow documentation master file, created by
   sphinx-quickstart on Thu May 28 16:38:36 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

About
=====================================

.. The user guide contains an introduction to the application and everything to get you started.


.. toctree::
   :maxdepth: 2
   :titlesonly:
   :hidden:

   self
   roadmap


Motivation and Scope
______________________________________________________

Main Costs of CV / ML / AI Projects

- Repetitive Data Handling
- **A lot** of trial and error
- Maintaining code snippets
- Providing a useful interface or low-code solution
- Communication and requirements engineering
- Labelling and results evaluation


Goal
______________________________________________________

Be user-friendly

- Modern, efficient UI
- stop re-writing the same code for easy, repetitive tasks

Avoid duplicate code across ML projects. Make code:

- Extensible
- Reusable

Train models for user-specified datasets

- Model architecture search
- Hyper-Parameter search
- Model inference

Automatically persist

- Workflows
- Settings
- Activity

Minimal on-boarding effort

- As simple as dropping in an excel list of file paths
- Write your own .py-based plugin files


Similar applications
______________________________________________________

Similar application are typically commercial cloud services.
Most of them offer on-premises for a high price.
The plugin and extension process has usually a high
on-boarding cost and learning curve.

Supervisely
^^^^^^^^^^^^^^

Supervisely tries to help labeling large datasets and integrate model training
and evaluation.

Main drawbacks are:

- The service is cloud based
- Closed source commercial.
- Limited support for data formats

Google Auto ML
^^^^^^^^^^^^^^

Google Auto ML tries to help managing large datasets and integrate model training
and evaluation. Also does hyper-parameter and model search.

Main drawbacks are:

- The service is cloud based
- Closed source and commercial.
- Limited support for arbitrary data formats. Datasets are required to be in a specific format.

LabelBox
^^^^^^^^^^^^^^

Automatically does preprocessing, has custom models.

- Cloud based service
- Closed source and commercial
- Limited support for arbitrary data formats. Datasets are required to be in a specific format.



