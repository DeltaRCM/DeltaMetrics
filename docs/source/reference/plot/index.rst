.. api.plot:

********************************
Plotting operations
********************************

The package uses a few utility classes and functions to make consistent plotting easy thoughout the package.
This reference page documents the lower-level utilities used to make this happen.

.. note::

    The built-in routines to plot ``Section`` and ``Plan`` objects are not documented here, look for documentation on those high-level methods in their respective module documentation.

.. hint::

  There is a complete :doc:`Visualization Guide </guides/subject_guides/visualization>`  about the organization of this area of sandplover, and examples for how to use and make visualizations.

The functions are defined in ``sandplover.plot``.

Plotting convenience functions
==============================

These functions may be helpful in making figures and exploring during analyses.
Mostly, these functions provide a component of a plot.

.. currentmodule:: sandplover.plot

.. autofunction:: aerial_view
.. autofunction:: overlay_sparse_array

.. autofunction:: style_axes_km
.. autofunction:: append_colorbar


sandplover plot routines
==========================

These functions are similar to the convenience functions above, but mostly produce their own plots entirely, rather than adding a component of a plot.

.. autosummary::
    :toctree: ../../_autosummary

    show_one_dimensional_trajectory_to_strata
    show_histograms


sandplover colormaps
======================

.. autofunction:: cartographic_colormap
.. autofunction:: aerial_colormap
.. autofunction:: vintage_colormap


Plotting utility functions
==========================

These functions are mostly used internally.

.. autofunction:: get_display_arrays
.. autofunction:: get_display_lines
.. autofunction:: get_display_limits
.. autofunction:: _fill_steps
.. autofunction:: _scale_lightness
