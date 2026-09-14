import copy
import sys

import netCDF4
import numpy as np
import pytest
import xarray as xr

from sandplover.io import DictionaryIO
from sandplover.io import NetCDFIO
from sandplover.sample_data.sample_data import _get_golf_path
from sandplover.sample_data.sample_data import _get_landsat_path

golf_path = _get_golf_path()
hdf_path = _get_landsat_path()


@pytest.fixture
def empty_netcdf_file(tmp_path):
    """Create blank NetCDF4 file."""
    p = tmp_path / "dummy.nc"
    f = netCDF4.Dataset(p, "w", format="NETCDF4")
    f.createVariable("test", "f4")
    f.close()
    return p


@pytest.fixture
def empty_txt_file(tmp_path):
    """Create a dummy text file."""
    p = tmp_path / "dummy.txt"
    p.touch()
    return p


class TestNetCDFIO:
    def test_netcdf_io_init(self):
        netcdf_io = NetCDFIO(golf_path)
        assert netcdf_io.io_type == "file"
        assert netcdf_io._engine == "netcdf4"
        assert len(netcdf_io._in_memory_variables) == 0

    def test_netcdf_io_keys(self):
        netcdf_io = NetCDFIO(golf_path)
        assert len(netcdf_io.keys) > 3

    def test_netcdf_io_nomemory(self):
        netcdf_io = NetCDFIO(golf_path)
        dataset_size = sys.getsizeof(netcdf_io.dataset)
        inmemory_size = sys.getsizeof(netcdf_io._in_memory_variables)

        var = "velocity"
        # slice the dataset directly
        velocity_arr = netcdf_io.dataset[var].data[:, 10, :]
        assert len(velocity_arr.shape) == 2
        assert type(velocity_arr) is np.ndarray

        dataset_size_after = sys.getsizeof(netcdf_io.dataset)
        inmemory_size_after = sys.getsizeof(netcdf_io._in_memory_variables)

        assert dataset_size == dataset_size_after
        assert inmemory_size == inmemory_size_after

    @pytest.mark.xfail()
    def test_netcdf_io_intomemory_direct(self):
        netcdf_io = NetCDFIO(golf_path, "netcdf")
        dataset_size = sys.getsizeof(netcdf_io.dataset)
        inmemory_size = sys.getsizeof(netcdf_io._in_memory_variables)

        var = "velocity"
        assert len(netcdf_io._in_memory_variables) == 0
        netcdf_io._in_memory_variables[var] = np.array(netcdf_io.dataset.variables[var])
        assert len(netcdf_io._in_memory_variables) == 1
        _arr = netcdf_io._in_memory_variables[var]

        dataset_size_after = sys.getsizeof(netcdf_io.dataset)
        inmemory_size_after = sys.getsizeof(netcdf_io._in_memory_variables)

        assert dataset_size == dataset_size_after
        assert inmemory_size < inmemory_size_after
        assert sys.getsizeof(_arr) > 1000

    @pytest.mark.xfail()
    def test_netcdf_io_intomemory_read(self):
        netcdf_io = NetCDFIO(golf_path, "netcdf")
        dataset_size = sys.getsizeof(netcdf_io.dataset)
        inmemory_size = sys.getsizeof(netcdf_io._in_memory_variables)

        var = "velocity"
        assert len(netcdf_io._in_memory_variables) == 0
        netcdf_io.read(var)
        assert len(netcdf_io._in_memory_variables) == 1
        _arr = netcdf_io._in_memory_variables[var]

        assert isinstance(_arr, xr.core.dataarray.DataArray)

        dataset_size_after = sys.getsizeof(netcdf_io.dataset)
        inmemory_size_after = sys.getsizeof(netcdf_io._in_memory_variables)

        assert dataset_size == dataset_size_after
        assert inmemory_size < inmemory_size_after

    def test_hdf5_io_init_without_engine(self):
        netcdf_io = NetCDFIO(hdf_path)
        assert netcdf_io.io_type == "file"
        assert netcdf_io._engine == "h5netcdf"
        assert len(netcdf_io._in_memory_variables) == 0

    def test_hdf5_io_init_with_engine(self):
        netcdf_io = NetCDFIO(hdf_path, engine="h5netcdf")
        assert netcdf_io.io_type == "file"
        assert netcdf_io._engine == "h5netcdf"
        assert len(netcdf_io._in_memory_variables) == 0

    def test_hdf5_io_keys(self):
        hdf5_io = NetCDFIO(hdf_path)
        assert len(hdf5_io.keys) == 7

    def test_nofile(self):
        with pytest.raises(FileNotFoundError):
            NetCDFIO("badpath")

    def test_empty_file(self, empty_netcdf_file):
        assert empty_netcdf_file.is_file()
        with pytest.raises(NotImplementedError):
            NetCDFIO(empty_netcdf_file)

    def test_invalid_file(self, empty_txt_file):
        assert empty_txt_file.is_file()
        with pytest.raises(TypeError):
            NetCDFIO(empty_txt_file)

    def test_readvar_intomemory(self):
        netcdf_io = NetCDFIO(golf_path, auxdata_path="meta")
        assert netcdf_io._in_memory_variables == {}

        netcdf_io.read("eta")
        assert ("eta" in netcdf_io._in_memory_variables) is True

    def test_readvar_intomemory_error(self):
        netcdf_io = NetCDFIO(golf_path)
        assert netcdf_io._in_memory_variables == {}

        with pytest.raises(KeyError):
            netcdf_io.read("nonexistant")

    def test_netcdf_no_metadata(self):
        # works fine, because there is no `connect` call in io init
        netcdf_io = NetCDFIO(golf_path)
        assert len(netcdf_io._in_memory_variables) == 0

    def test_register_variable(self):
        # this private method assumes shape already validated by cube, so we
        # just verify here that the method is callable
        netcdf_io = NetCDFIO(golf_path)
        assert len(netcdf_io._in_memory_variables) == 0
        netcdf_io._register_variable("test", np.zeros((100, 100)))
        assert len(netcdf_io._in_memory_variables) == 1
        assert "test" in netcdf_io.known_variables


class TestDictionaryIO:
    _shape = (50, 100, 200)
    dict_xr = {"eta": xr.DataArray(np.random.normal(size=_shape))}
    dict_np = {
        "eta": np.random.normal(size=_shape),
        "velocity": np.random.normal(size=_shape),
    }

    def test_create_from_xarray_data(self):
        dict_io = DictionaryIO(self.dict_xr)
        assert ("eta" in dict_io._in_memory_variables) is True
        assert isinstance(dict_io["eta"], xr.core.dataarray.DataArray)

    def test_create_with_auxdata(self):
        _dict_w_aux = copy.deepcopy(self.dict_np)
        _dict_w_aux["auxdata"] = self.dict_np
        dict_io_str = DictionaryIO(_dict_w_aux, auxdata_path="auxdata")
        dict_io_dict = DictionaryIO(self.dict_np, auxdata_path=self.dict_np)
        assert isinstance(dict_io_str["eta"], np.ndarray)
        assert isinstance(dict_io_str.aux["eta"], np.ndarray)
        assert isinstance(dict_io_dict["eta"], np.ndarray)
        assert isinstance(dict_io_dict.aux["eta"], np.ndarray)

    def test_dimensions_ignored_if_xarray(self):
        dict_io = DictionaryIO(self.dict_xr, dimensions=(3, 4, 5))
        assert ("eta" in dict_io._in_memory_variables) is True
        assert isinstance(dict_io["eta"], xr.core.dataarray.DataArray)

    def test_create_from_numpy_data_nodims(self):
        dict_io = DictionaryIO(self.dict_np)
        assert ("eta" in dict_io._in_memory_variables) is True
        assert ("velocity" in dict_io._in_memory_variables) is True
        assert isinstance(dict_io["eta"], np.ndarray)
        assert isinstance(dict_io["dim0"], np.ndarray)

    def test_create_from_numpy_data_dimensions(self):
        dict_io = DictionaryIO(
            self.dict_np,
            dimensions={
                "time": np.arange(self._shape[0]),
                "x": np.arange(self._shape[1]),
                "y": np.arange(self._shape[2]),
            },
        )
        assert ("eta" in dict_io._in_memory_variables) is True
        assert ("velocity" in dict_io._in_memory_variables) is True
        assert isinstance(dict_io["eta"], np.ndarray)
        assert isinstance(dict_io["time"], np.ndarray)
        assert isinstance(dict_io["x"], np.ndarray)
        assert isinstance(dict_io["y"], np.ndarray)
        assert np.all(dict_io["eta"] == self.dict_np["eta"])

    def test_bad_dimensions_types(self):
        with pytest.raises(TypeError, match=r".* type for `dimensions` .*"):
            _ = DictionaryIO(self.dict_np, dimensions=(3, 4, 5))
        with pytest.raises(TypeError, match=r".* type for `dimensions` .*"):
            _ = DictionaryIO(self.dict_np, dimensions=1)
        with pytest.raises(TypeError, match=r".* type for `dimensions` .*"):
            _ = DictionaryIO(self.dict_np, dimensions="string")
        with pytest.raises(TypeError, match=r".* type for `dimensions` .*"):
            _ = DictionaryIO(self.dict_np, dimensions=["list", "string"])

    def test_bad_dimensions_length(self):
        with pytest.raises(ValueError, match=r"`dimensions` must .*"):
            _ = DictionaryIO(self.dict_np, dimensions={})

    def test_bad_dimensions_shape_mismatch(self):
        with pytest.raises(ValueError, match=r"Shape of `dimensions` .*"):
            # note dim2 and dim1 are switchd below!
            DictionaryIO(
                self.dict_np,
                dimensions={
                    "time": np.arange(self._shape[0]),
                    "x": np.arange(self._shape[2]),
                    "y": np.arange(self._shape[1]),
                },
            )

    def test_not_implemented_methods(self):
        dict_io = DictionaryIO(self.dict_xr)
        with pytest.raises(NotImplementedError):
            dict_io.connect()
        with pytest.raises(NotImplementedError):
            dict_io.read()
        with pytest.raises(NotImplementedError):
            dict_io.write()

    def test_dict_1d_first_with_dimensions_works(self):
        """Leading 1-D var must not poison dims validation."""
        shape = (10, 50, 60)
        dims = {
            "t_ax": np.arange(shape[0]),
            "y_ax": np.arange(shape[1]),
            "x_ax": np.arange(shape[2]),
        }
        data = {
            "station_id": np.arange(100),  # 1-D comes first
            "temperature": np.random.rand(*shape),  # 3-D ref var
        }
        dict_io = DictionaryIO(data, dimensions=dims)  # should not raise

        assert dict_io.dims == list(dims.keys())
        for k in dims:
            assert np.array_equal(dict_io[k], np.asarray(dims[k]))
        assert ("station_id" in dict_io._in_memory_variables) is True
        assert dict_io["temperature"].shape == shape

    def test_dict_no_3d_with_dimensions_works(self):
        """No 3-D vars present, but explicit dimensions → accept."""
        dims = {"t_ax": np.arange(2), "y_ax": np.arange(3), "x_ax": np.arange(4)}
        data = {"station_id": np.arange(100), "quality": np.arange(50)}  # only 1-D
        dict_io = DictionaryIO(data, dimensions=dims)  # should not raise

        assert dict_io.dims == list(dims.keys())
        for k in dims:
            assert len(dict_io[k]) == len(dims[k])
        assert set(data.keys()).issubset(set(dict_io._in_memory_variables.keys()))

    def test_dict_no_3d_no_dimensions_errors(self):
        """No 3-D vars and no dimensions → clear error."""
        data = {"station_id": np.arange(100), "quality": np.arange(50)}  # only 1-D
        with pytest.raises(ValueError, match=r"Cannot infer coordinates"):
            _ = DictionaryIO(data)  # must raise

    def test_register_variable(self):
        # this private method assumes shape already validated by cube, so we
        # just verify here that the method is callable
        dict_io = DictionaryIO(self.dict_xr)
        assert ("eta" in dict_io._in_memory_variables) is True
        assert len(dict_io._in_memory_variables) == 1
        dict_io._register_variable("test", np.zeros((100, 100)))
        assert len(dict_io._in_memory_variables) == 2
        assert "test" in dict_io.known_variables
