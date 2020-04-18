
import abc
import os
import sys

import numpy as np
import netCDF4


def known_variables():
    return ['eta', 'stage', 'depth', 'discharge',
            'velocity', 'strata_age', 'strata_sand_frac', 'strata_depth']


def known_coords():
    return ['x', 'y', 'time']


class Base_IO(abc.ABC):
    """BaseIO object other file format wrappers inheririt from.

    .. note::
        This is an abstract class and cannot be instantiated directly. If you
        wish to subclass to create a new IO format, you must implement the
        following methods: ``load``, .
    """

    def __init__(self, data_path, write):
        """Initialize the base IO.
        """
        self.known_variables = known_variables()
        self.known_coords = known_coords()

        self.data_path = data_path
        self.write = write

        self.connect()

    @property
    def data_path(self):
        """`str` : Path to data file.

        Parameters
        ----------
        data_path : str
            path to data file for IO operations.
        """
        return self._data_path

    @data_path.setter
    def data_path(self, var):
        self._data_path = var

    @abc.abstractmethod
    def connect(self):
        """Should connect to the data file.

        This function should initialize the file if it does not exist, or
        connect to the file if it already exists---but *do not* read the file.
        """
        return

    @abc.abstractmethod
    def read(self):
        """Should read data into memory.
        """
        return

    @abc.abstractmethod
    def write(self):
        """Should write the data to file.

        Take a :obj:`~deltametrics.cube.Cube` and write it to file.
        """
        return

    @abc.abstractmethod
    def __getitem__(self):
        """Should slice the data from file.
        """
        return

    @property
    @abc.abstractmethod
    def keys(self):
        """Should link to all key _names_ stored in file.
        """
        return


class NetCDF_IO(Base_IO):
    """Utility for consistent IO with netCDF files.

    This module wraps calls to the netCDF4 python module in a consistent API,
    so the user can work seamlessly with either netCDF4 files or HDF5 files.
    The public methods of this class are consistent with
    :obj:`~deltametrics.utils.HDF_IO`.
    """

    def __init__(self, data_path, write=False):
        """Initialize the NetCDF_IO handler.

        Parameters
        ----------
        data_path : `str`
            Path to file to read or write to.

        load : `bool`
            Whether to load the file into memory

        write : `bool`, optional
            Whether to allow writing to an existing file. Set to False by
            default, if a file already exists at ``data_path``, writing is
            disabled, unless ``write`` is set to True.

        """
        super().__init__(data_path=data_path, write=write)

        self.type = 'netcdf'

        self._in_memory_data = {}

    def connect(self):
        """Connect to the data file.

        Initialize the file if it does not exist, or simply ``return`` if the
        file already exists.

        .. note::
            This function is automatically called during initialization of a
            :obj:`~deltametrics.cube.Cube`.
        """
        if not os.path.isfile(self.data_path):
            self.dataset = netCDF4.Dataset(
                self.data_path, "w", format="NETCDF4")
        else:
            if self.write:
                self.dataset = netCDF4.Dataset(self.data_path, "r+")
            else:
                self.dataset = netCDF4.Dataset(self.data_path, "r")

    def __read(self, name=None):
        """Read variables from file and into memory.

        Convert `variables` in netCDF file to `ndarray`.

        This operation is usually used in preparation for coersion into a
        :obj:`~deltametrics.cube.Cube` instance that is loaded into memory.

        Parameters
        ----------
        name : :obj:`list` of :obj:`str`, `str`, optional
            Which variables to load from the file. Default is to load all
            variables.

        """
        _data = _DataDict({})
        _coord = _DataDict({})
        if name:
            load_list = name
        else:
            load_list = self.dataset.variables

        for var in load_list:
            try:
                _arr = self.dataset.variables[var]
            except ValueError as e:
                raise e
            if var in self.known_coords:
                _coord[var] = np.array(_arr)
            else:
                _data[var] = np.array(_arr)

        return _data, _coord

    def read(self, var):
        """Read variable from file and into memory.

        Converts `variables` in netCDF file to `ndarray` for coersion into a
        :obj:`~deltametrics.cube.Cube` instance.

        Parameters
        ----------
        var : `str`
            Which variable to load from the file.

        """
        try:
            _arr = self.dataset.variables[var]
        except ValueError as e:
            raise e
        self._in_memory_data[var] = np.array(_arr)

    def write(self):
        """Write the data to file.

        Take a :obj:`~deltametrics.cube.Cube` and write it to file.
        """
        raise NotImplementedError

    def __getitem__(self, var):
        if var in self._in_memory_data.keys():
            return self._in_memory_data[var]
        else:
            return self.dataset.variables[var]

    @property
    def keys(self):
        """Link to variable names in file.
        """
        return [var for var in self.dataset.variables]


class HDF_IO(Base_IO):
    """Utility for consistent IO with HDF5 files.

    This module wraps calls to the hdf5 python module in a consistent API,
    so the user can work seamlessly with either HDF5 files or netCDF4 files.
    The public methods of this class are consistent with
    :obj:`~deltametrics.utils.NetCDF_IO`.
    """

    def __init__(self, data_path):
        """Initialize the HDF5_IO handler

        parameters
        """
        raise NotImplementedError
