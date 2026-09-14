***************
Cube operations
***************

The package makes available `Cube` objects, which are the central office to all of the other functionality of sandplover.

The cubes keep track of underlying data, which may represent any number of unique variables.
For example, the :obj:`~sandplover.cube.DataCube` connects to a set of data in the form of a ``t-x-y`` array-like dataset, and associated auxiliary data defining the array coordinates, type, and data units.
So, variables in the underlying data might be lidar scans, overhead photos, grain size maps (pyDeltaRCM), or flow velocity records (pyDeltaRCM), etc.

The functions are defined in ``sandplover.cube``.


Cube classes
==============

.. currentmodule:: sandplover.cube

.. autosummary::
    :toctree: ../../_autosummary

    DataCube
        :special-members:
    StratigraphyCube
        :special-members:
    BaseCube
        :special-members:
