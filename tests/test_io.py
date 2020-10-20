import pytest

import sys
import os

import numpy as np
import xarray as xr

from deltametrics import io
import utilities

rcm8_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                         'sample_data', 'files', 'pyDeltaRCM_Output_8.nc')

hdf_path = os.path.join(os.path.dirname(__file__), '..', 'deltametrics',
                        'sample_data', 'files',
                        'LandsatEx.hdf5')


def test_netcdf_io_init():
    netcdf_io = io.NetCDFIO(rcm8_path, 'netcdf')
    assert netcdf_io.type == 'netcdf'
    assert len(netcdf_io._in_memory_data.keys()) == 0


def test_netcdf_io_keys():
    netcdf_io = io.NetCDFIO(rcm8_path, 'netcdf')
    assert len(netcdf_io.keys) == 11


def test_netcdf_io_nomemory():
    netcdf_io = io.NetCDFIO(rcm8_path, 'netcdf')
    dataset_size = sys.getsizeof(netcdf_io.dataset)
    inmemory_size = sys.getsizeof(netcdf_io._in_memory_data)

    var = 'velocity'
    # slice the dataset directly
    velocity_arr = netcdf_io.dataset[var].data[:, 10, :]
    assert len(velocity_arr.shape) == 2
    assert type(velocity_arr) is np.ndarray

    dataset_size_after = sys.getsizeof(netcdf_io.dataset)
    inmemory_size_after = sys.getsizeof(netcdf_io._in_memory_data)

    assert dataset_size == dataset_size_after
    assert inmemory_size == inmemory_size_after


@pytest.mark.xfail()
def test_netcdf_io_intomemory_direct():
    netcdf_io = io.NetCDFIO(rcm8_path, 'netcdf')
    dataset_size = sys.getsizeof(netcdf_io.dataset)
    inmemory_size = sys.getsizeof(netcdf_io._in_memory_data)

    var = 'velocity'
    assert len(netcdf_io._in_memory_data.keys()) == 0
    netcdf_io._in_memory_data[var] = np.array(netcdf_io.dataset.variables[var])
    assert len(netcdf_io._in_memory_data.keys()) == 1
    _arr = netcdf_io._in_memory_data[var]

    dataset_size_after = sys.getsizeof(netcdf_io.dataset)
    inmemory_size_after = sys.getsizeof(netcdf_io._in_memory_data)

    assert dataset_size == dataset_size_after
    assert inmemory_size < inmemory_size_after
    assert sys.getsizeof(_arr) > 1000


@pytest.mark.xfail()
def test_netcdf_io_intomemory_read():
    netcdf_io = io.NetCDFIO(rcm8_path, 'netcdf')
    dataset_size = sys.getsizeof(netcdf_io.dataset)
    inmemory_size = sys.getsizeof(netcdf_io._in_memory_data)

    var = 'velocity'
    assert len(netcdf_io._in_memory_data.keys()) == 0
    netcdf_io.read(var)
    assert len(netcdf_io._in_memory_data.keys()) == 1
    _arr = netcdf_io._in_memory_data[var]

    assert isinstance(_arr, xr.core.dataarray.DataArray)

    dataset_size_after = sys.getsizeof(netcdf_io.dataset)
    inmemory_size_after = sys.getsizeof(netcdf_io._in_memory_data)

    assert dataset_size == dataset_size_after
    assert inmemory_size < inmemory_size_after


def test_hdf5_io_init():
    netcdf_io = io.NetCDFIO(hdf_path, 'hdf5')
    assert netcdf_io.type == 'hdf5'
    assert len(netcdf_io._in_memory_data.keys()) == 0


def test_hdf5_io_keys():
    hdf5_io = io.NetCDFIO(hdf_path, 'hdf5')
    assert len(hdf5_io.keys) == 7


def test_nofile():
    with pytest.raises(FileNotFoundError):
        io.NetCDFIO('badpath', 'netcdf')


def test_empty_file(tmp_path):
    p = utilities.create_dummy_netcdf(tmp_path)
    with pytest.warns(UserWarning):
        io.NetCDFIO(p, 'netcdf')


def test_invalid_file(tmp_path):
    p = utilities.create_dummy_txt_file(tmp_path)
    with pytest.raises(TypeError):
        io.NetCDFIO(p, 'netcdf')


def test_readvar_intomemory():
    netcdf_io = io.NetCDFIO(rcm8_path, 'netcdf')
    assert netcdf_io._in_memory_data == {}

    netcdf_io.read('eta')
    assert ('eta' in netcdf_io._in_memory_data.keys()) is True


def test_readvar_intomemory_error():
    netcdf_io = io.NetCDFIO(rcm8_path, 'netcdf')
    assert netcdf_io._in_memory_data == {}

    with pytest.raises(KeyError):
        netcdf_io.read('nonexistant')
