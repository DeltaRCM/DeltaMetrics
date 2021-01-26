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
    nc_bool = [os.path.splitext(fname)[1] == '.nc' for fname in fnames]
    nc_idx = [i for i, b in enumerate(nc_bool) if b]
    golf_path = fnames[nc_idx[0]]
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
        nt = 5
        ts = np.linspace(0, golf['eta'].shape[0]-1, num=nt, dtype=np.int)  # linearly interpolate ts

        fig, ax = plt.subplots(1, nt, figsize=(12, 2))
        for i, t in enumerate(ts):
            ax[i].imshow(golf['eta'][t, :, :], vmin=-2, vmax=0.5) 
            ax[i].set_title('t = ' + str(t)) 
            ax[i].axes.get_xaxis().set_ticks([]) 
            ax[i].axes.get_yaxis().set_ticks([]) 
        ax[0].set_ylabel('y-direction') 
        ax[0].set_xlabel('x-direction') 
        plt.show() 
    """
    golf_path = _get_golf_path()
    return cube.DataCube(golf_path)


def tdb12():
    raise NotImplementedError


def _get_rcm8_path():
    rcm8_path = REGISTRY.fetch('pyDeltaRCM_Output_8.nc')
    return rcm8_path


def rcm8():
    """Rcm8 Delta dataset.

    This is a synthetic delta dataset generated from the pyDeltaRCM numerical
    model. Unfortunately, we do not know the specific version of pyDeltaRCM
    the model run was executed with. Moreover, many new coupling features have
    been added to pyDeltaRCM and DeltaMetrics since this run. As a result,
    this dataset is slated to be deprecated at some point, in favor of the
    :obj:`golf` dataset. 

    If you are learning to use DeltaMetrics or developing new codes or
    documentation, please use the :obj:`golf` delta dataset. 

    .. plot::

        rcm8 = dm.sample_data.rcm8()
        nt = 5
        ts = np.linspace(0, rcm8['eta'].shape[0]-1, num=nt, dtype=np.int)  # linearly interpolate ts

        fig, ax = plt.subplots(1, nt, figsize=(12, 2))
        for i, t in enumerate(ts):
            ax[i].imshow(rcm8['eta'][t, :, :], vmin=-2, vmax=0.5) 
            ax[i].set_title('t = ' + str(t)) 
            ax[i].axes.get_xaxis().set_ticks([]) 
            ax[i].axes.get_yaxis().set_ticks([]) 
        ax[0].set_ylabel('y-direction') 
        ax[0].set_xlabel('x-direction') 
        plt.show()
    """
    rcm8_path = _get_rcm8_path()
    return cube.DataCube(rcm8_path)


def _get_landsat_path():
    landsat_path = REGISTRY.fetch('LandsatEx.hdf5')
    return landsat_path


def landsat():
    """Landsat image dataset.

    This is a set of satellite images from the Landsat 5 satellite, collected
    over the Krishna River delta, India. The dataset includes annual-composite
    scenes from four different years (`[1995, 2000, 2005, 2010]`) and includes
    data collected from four bands (`['Red', 'Green', 'Blue', 'NIR']`).

    .. plot::

        landsat = dm.sample_data.landsat()
        nt = landsat.shape[0]

        maxr = np.max(landsat['Red'][:])
        maxg = np.max(landsat['Green'][:])
        maxb = np.max(landsat['Blue'][:])

        fig, ax = plt.subplots(1, nt, figsize=(12, 2))
        for i in np.arange(nt):
            _arr = np.dstack((landsat['Red'][i, :, :]/maxr,
                              landsat['Green'][i, :, :]/maxg,
                              landsat['Blue'][i, :, :]/maxb))
            ax[i].imshow(_arr)
            ax[i].set_title('year = ' + str(landsat.t[i]))
            ax[i].axes.get_xaxis().set_ticks([])
            ax[i].axes.get_yaxis().set_ticks([])

        plt.show()
    """
    landsat_path = _get_landsat_path()
    return cube.DataCube(landsat_path)
