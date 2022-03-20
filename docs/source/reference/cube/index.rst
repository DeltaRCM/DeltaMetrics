***************
Cube operations
***************

The package makes available `Cube` objects, which are the central office to all of the other functionality of DeltaMetrics. 

The cubes keep track of underlying data, which may represent any number of unique variables. 
For example, the :obj:`~deltametrics.cube.DataCube` connects to a set of data in the form of a ``t-x-y`` array-like dataset, and associated metadata defining the array coordinates, type, and data units. 
So, variables in the underlying data might be lidar scans, overhead photos, grain size maps (pyDeltaRCM), or flow velocity records (pyDeltaRCM), etc.

The functions are defined in ``deltametrics.cube``. 


Cube classes
==============

.. currentmodule:: deltametrics.cube

.. autosummary:: 
    :toctree: ../../_autosummary

    DataCube
        :special-members:
    StratigraphyCube
        :special-members:
    BaseCube
        :special-members:
