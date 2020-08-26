.. api.settle:

*********************************
Stratigraphy functions
*********************************

The package makes available functions relating to computing things based on
stratigraphy.

The functions are defined in ``deltametrics.strat``. 


Compute stratigraphy routines
=============================

.. currentmodule:: deltametrics.strat

.. autosummary::
    :toctree: ../../_autosummary

    compute_boxy_stratigraphy_volume
    compute_boxy_stratigraphy_coordinates


Stratigraphy statistics functions
=================================

.. autosummary::
    :toctree: ../../_autosummary

    compute_trajectory
    compute_compensation


Quick-stratigraphy attributes classes
=====================================

The "quick" stratigraphy attributes provide a common API that is accessed by
`DataCube`, `DataSectionVariable` and `DataPlanformVariable` methods. There
are two methods of computing quick stratigraphy, with
:obj:`~deltametrics.strat.MeshStratigraphyAttributes` as the default.

.. autosummary::
    :toctree: ../../_autosummary

    MeshStratigraphyAttributes
    BoxyStratigraphyAttributes
    BaseStratigraphyAttributes


Low-level stratigraphy utility functions
========================================

The functions outlined in this section are the main functions that do the actual work of computing stratigraphy and preservation, throughout DeltaMetrics. These functions may be useful if you are trying to use parts of DeltaMetrics, but need to customize something for your own use-case.

.. autofunction:: _compute_elevation_to_preservation
.. autofunction:: _compute_preservation_to_time_intervals
.. autofunction:: _compute_preservation_to_cube
.. autofunction:: _determine_strat_coordinates
