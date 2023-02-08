.. api.settle:

*********************************
Stratigraphy operations
*********************************

The package makes available functions relating to computing things based on
stratigraphy.

The functions are defined in ``deltametrics.strat``. 

.. currentmodule:: deltametrics.strat


Stratigraphy statistics functions
=================================

Metrics and statistics of stratigraphy.

.. autosummary::
    :toctree: ../../_autosummary

    compute_compensation
    compute_net_to_gross
    compute_thickness_surfaces
    compute_sedimentograph


Compute stratigraphy routines
=============================

Routines to compute stratigraphic volumes.

.. autosummary::
    :toctree: ../../_autosummary

    compute_boxy_stratigraphy_volume
    compute_boxy_stratigraphy_coordinates


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
.. autofunction:: _adjust_elevation_by_subsidence
.. autofunction:: _determine_deposit_from_background