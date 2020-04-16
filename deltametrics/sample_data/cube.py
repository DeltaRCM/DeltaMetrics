"""Sample data cubes

A sample of a data cube (x,y,t) for examples, tests, etc.

Example
-------

A Data cube can be instantiated as, for example::

  >>> import deltametrics as dm
  >>> _tulane = dm.sample_data.cube.tdb12()

Available information on the data cubes is enumerated in the following
section.


Example data cubes
------------------------

:meth:`tdb12` : `ndarray`
  This data cube is from Tulane Delta Basin expt 12. 

:meth:`rcm1` : `ndarray`
  This data cube is from the DeltaRCM model. 

"""

import sys, os

import numpy as np
import netCDF4

from ..cube import Cube
from ..utils import NetCDF_IO, HDF_IO



def tdb12():
    raise NotImplementedError
    return np.array([[]])


def rcm8():
    path = os.path.join(os.path.dirname(__file__), 'files', 'Output_8', 'pyDeltaRCM_output.nc')
    cube = Cube(path)
    return cube

