.. api.plot:

********************************
Plotting utilities and functions
********************************

The package uses a few utility classes and functions to make consistent plotting easy thoughout the package.
This reference page documents the lower-level utilities used to make this happen.

.. note::
    The built-in routines to plot ``Section`` and ``Plan`` objects are not documented here, look for documentation on those high-level methods in their respective module documentation.

The functions are defined in ``deltametrics.plot``. 

Default styling
===============

By default, each variable receives a set of styling definitions.
The default parameters of each styling variable are defined below:

.. plot:: plot/document_variableset.py


Plotting utility objects
========================

.. currentmodule:: deltametrics.plot

.. autosummary:: 
    :toctree: ../../_autosummary

    VariableInfo
    VariableSet


Plotting utility functions
==========================

.. autofunction:: append_colorbar


DeltaMetrics colormaps
======================

.. autofunction:: cartographic_colormap
.. autofunction:: aerial_colormap
