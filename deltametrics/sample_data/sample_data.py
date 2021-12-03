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
    model. This model run was created to generate sample data. Model was run
    on 10/14/2021, at the University of Texas at Austin.

    Run was computed with pyDeltaRCM v2.1.0. See log file for complete
    information on system and model configuration.

    Data available at Zenodo, https://doi.org/10.5281/zenodo.4456143.

    Version history:
    v1.1: 10.5281/zenodo.5570962
    v1.0: 10.5281/zenodo.4456144

    .. plot::

        golf = dm.sample_data.golf()
        nt = 5
        ts = np.linspace(0, golf['eta'].shape[0]-1, num=nt, dtype=np.int)

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
    return cube.DataCube(golf_path, coordinates={'x': 'y', 'y': 'x'})


def tdb12():
    raise NotImplementedError


def _get_aeolian_path():
    aeolian_path = REGISTRY.fetch('swanson_aeolian_expt1.nc')
    return aeolian_path


def aeolian():
    """An aeolian dune field dataset.

    This is a synthetic delta dataset generated from the Swanson et al.,
    2017 "A Surface Model for Aeolian Dune Topography" numerical model. The
    data have been subsetted, only keeping the first 500 saved timesteps, and
    formatted into a netCDF file.

    Swanson, T., Mohrig, D., Kocurek, G. et al. A Surface Model for Aeolian
    Dune Topography. Math Geosci 49, 635–655
    (2017). https://doi.org/10.1007/s11004-016-9654-x

    dataset reference: https://doi.org/10.6084/m9.figshare.17118827.v1

    Details:
        * default simualtion parameters were used.
        * only the first 500 timesteps of the simulation were recorded into
          the netcdf file.
        * the ordering for "easting" and "northing" coordinates in the netCDF
          file is opposite from the paper---the data are the same, but the
          source region is along the second axis; thus the display is
          different from the original paper.
        * simulation used the model code included as a supplement to the paper
          found here:
          https://static-content.springer.com/esm/art:10.1007/s11004-016-9654-x/MediaObjects/11004_2016_9654_MOESM5_ESM.txt
        * simulation was executed on 12/02/2021 with Matlab R2021a on Ubuntu
          20.04.

    .. plot::

        aeolian = dm.sample_data.aeolian()
        nt = 5
        ts = np.linspace(0, aeolian['eta'].shape[0]-1, num=nt, dtype=np.int)

        fig, ax = plt.subplots(1, nt, figsize=(7, 5))
        for i, t in enumerate(ts):
            ax[i].imshow(aeolian['eta'][t, :, :], vmin=-4, vmax=5)
            ax[i].set_title('t = ' + str(t))
            ax[i].axes.get_xaxis().set_ticks([])
            ax[i].axes.get_yaxis().set_ticks([])
        plt.show()
    """
    aeolian_path = _get_aeolian_path()
    return cube.DataCube(aeolian_path)


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

    .. important::
        
        If you are learning to use DeltaMetrics or developing new codes or
        documentation, please use the :obj:`golf` delta dataset.

    .. warning:: This cube may be removed in future releases.

    .. plot::

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rcm8 = dm.sample_data.rcm8()
        nt = 5
        ts = np.linspace(0, rcm8['eta'].shape[0]-1, num=nt, dtype=np.int)

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
