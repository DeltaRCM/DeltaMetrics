.. api.observation:

*********************************
The data Cube
*********************************

The package makes available a `Cube` object, which is the central office to all of the other functionality of DeltaMetrics. These data cubes can hold any single, or many attributes, for an ``x-y-v`` `ndarray` a value specifies the type of data stored in ``v``. ``v`` could be, lidar scans, overhead photos, grain size maps (DeltaRCM), flow velocity (DeltaRCM), etc.

The functions are defined in ``deltametrics.cube``. 


Cube types
==============

.. currentmodule:: deltametrics.cube

.. autosummary:: 
    :toctree: ../../_autosummary

    DataCube
        :special-members:
    BaseCube
    	:special-members:


Cube returns
===============

.. autosummary:: 
    :toctree: ../../_autosummary

	CubeVariable
		:special-members:
