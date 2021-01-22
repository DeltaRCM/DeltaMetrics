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
import pkg_resources
import warnings

import numpy as np
import netCDF4
import pooch

from .. import cube
from .. import utils


# deltametrics version
__version__ = utils._get_version()

# enusre DeprecationWarning is shown
warnings.simplefilter("default")


# configure the data registry
REGISTRY = pooch.create(
    path=pooch.os_cache("deltametrics"),
    base_url='https://github.com/DeltaRCM/DeltaMetrics/raw/develop/deltametrics/sample_data/files/',
    env="DELTAMETRICS_DATA_DIR",
)
with pkg_resources.resource_stream("deltametrics.sample_data", "registry.txt") as registry_file:
    REGISTRY.load_registry(registry_file)


def _get_golf_path():
    unpack = pooch.Unzip()
    fnames = REGISTRY.fetch('golf.zip', processor=unpack)
    golf_path = fnames[0]
    return golf_path


def golf():
    """Golf Delta dataset.

    This is a synthetic delta dataset generated from the pyDeltaRCM numerical
    model. The data were generated as one job in an ensemble executed
    2021-01-15 on the TACC supercomputer at the University of Texas at Austin.

    Run was computed with pyDeltaRCM v1.1.1, while at commit hash
    58244313796273ca4eeb8ea8d724884dd51510a1.

    Data available as Zenodo doi 10.5281/zenodo.4456144

    .. plot::
        golf = dm.sample_data.golf()
        fig, ax = plt.subplots()
        golf.show_plan(t=-1, ax=ax)
        plt.show()

    .. note::

        Data is handled by `pooch` and will be downloaded and cached on local
        computer as needed.
    """
    golf_path = _get_golf_path()
    return cube.DataCube(golf_path)


def tdb12():
    raise NotImplementedError


def _get_rcm8_path():
    rcm8_path = REGISTRY.fetch('pyDeltaRCM_Output_8.nc')
    return rcm8_path


def rcm8():
    """A sample pyDeltaRCM file, as netCDF.

    .. note::

        Data is handled by `pooch` and will be downloaded and cached on local
        computer as needed.
    """
    rcm8_path = _get_rcm8_path()
    return cube.DataCube(rcm8_path)


def _get_landsat_path():
    landsat_path = REGISTRY.fetch('LandsatEx.hdf5')
    return landsat_path


def landsat():
    """A sample Landsat image of a delta.
    """
    landsat_path = _get_landsat_path()
    return cube.DataCube(landsat_path)
