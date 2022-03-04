.. deltametrics documentation master file

************
DeltaMetrics
************

.. image:: https://github.com/DeltaRCM/DeltaMetrics/workflows/build/badge.svg
  :target: https://github.com/DeltaRCM/DeltaMetrics/actions

.. image:: https://codecov.io/gh/DeltaRCM/DeltaMetrics/branch/develop/graph/badge.svg
  :target: https://codecov.io/gh/DeltaRCM/DeltaMetrics

DeltaMetrics is a Python package for manipulating depositional system data cubes.
In particular, DeltaMetrics has robust objects and routines designed to help organize, visualize, and analyze topography and sedimentological timeseries data deltaic systems (e.g., the pyDeltaRCM numerical model and laboratory delta experiments).

.. plot:: guides/cover.py

   A :obj:`~deltametrics.plan.Planform` view of bed elevation in a modeled deltaic deposit, and a cross-:obj:`~deltametrics.section.StrikeSection` view of sediment deposition timing in stratigraphy.


DeltaMetrics documentation
#########################################


.. image:: https://img.shields.io/static/v1?label=GitHub&logo=github&message=source&color=brightgreen
    :target: https://github.com/DeltaRCM/DeltaMetrics

.. image:: https://badge.fury.io/gh/DeltaRCM%2FDeltaMetrics.svg
    :target: https://github.com/DeltaRCM/DeltaMetrics/releases

Documentation Table of Contents
-------------------------------

.. toctree::
   :maxdepth: 1

   meta/installing
   meta/contributing
   meta/license
   meta/conduct
   meta/planning


.. toctree::
   :maxdepth: 2

   guides/getting_started
   guides/userguide
   guides/devguide
   guides/visualization
   reference/index


Project status
##############

DeltaMetrics is currently in development, so the API is not *guaranteed* to be stable between versions; we do make an effort to maintain backwards compatibility. 
This open-source software is made available without warranty or guarantee of accuracy or correctness.
