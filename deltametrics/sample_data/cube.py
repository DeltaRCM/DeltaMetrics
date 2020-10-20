"""Sample data cubes

A sample of a data cube (x,y,t) for examples, tests, etc.

Example
-------

A Data cube can be instantiated as, for example::

.. doctest::

    >>> import deltametrics as dm
    >>> rcm8cube = dm.sample_data.cube.rcm8()
    >>> rcm8cube
    <deltametrics.cube.DataCube at 0x...>


Available information on the data cubes is enumerated in the following
section.


Example data cubes
------------------------

:meth:`tdb12` : `ndarray`
    This data cube is from Tulane Delta Basin expt 12.

:meth:`rcm8` : `ndarray`
    This data cube is from the pyDeltaRCM model.

"""

import sys
import os

import numpy as np
import netCDF4

from .. import cube
from ..io import NetCDFIO


def tdb12():
    raise NotImplementedError
    return np.array([[]])


def rcm8():
    """A sample pyDeltaRCM file, as netCDF.
    """
    path = os.path.join(os.path.dirname(__file__), 'files',
                        'pyDeltaRCM_Output_8.nc')
    _cube = cube.DataCube(path)
    return _cube
