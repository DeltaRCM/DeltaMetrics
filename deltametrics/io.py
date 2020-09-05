
import abc
import os
import sys

import numpy as np
import xarray as xr
import netCDF4


def known_variables():
    """A list of known variables.

    These variables are common variables we anticipate being present in all
    sorts of use-cases. Each one is given a set of default parameters in
    :obj:`~deltametrics.plot.VariableSet`.
    """
    return ['eta', 'stage', 'depth', 'discharge',
            'velocity', 'strata_sand_frac']


def known_coords():
    """A list of known coordinates.

    These coordinates are commonly defined coordinate matricies that may be
    stored inside of a file on disk. We don't treat these any differently in
    the io wrappers, but knowing they are coordinates can be helpful.
    """

    return ['x', 'y', 'time']


class BaseIO(abc.ABC):
    """BaseIO object other file format wrappers inheririt from.

    .. note::
        This is an abstract class and cannot be instantiated directly. If you
        wish to subclass to create a new IO format, you must implement the
        methods ``connect``, ``read``, ``write``, and ``keys``.
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

        Notes
        -----
        The setter method validates the path, and returns a ``FileNotFoundError`` if
        the file is not found.
        """
        return self._data_path

    @data_path.setter
    def data_path(self, var):
        if os.path.exists(var):
            self._data_path = var
        else:
            raise FileNotFoundError('File not found at supplied path: %s' % var)

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


class NetCDFIO(BaseIO):
    """Utility for consistent IO with netCDF files.

    This module wraps calls to the netCDF4 python module in a consistent API,
    so the user can work seamlessly with either netCDF4 files or HDF5 files.
    The public methods of this class are consistent with
    :obj:`~deltametrics.utils.HDFIO`.
    """

    def __init__(self, data_path, write=False):
        """Initialize the NetCDFIO handler.

        Initialize a connection to a NetCDF file.

        Parameters
        ----------
        data_path : `str`
            Path to file to read or write to.

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
        file already exists. This connection to the data file is "lazy"
        loading, meaning that array values are not being loaded into memory.

        .. note::
            This function is automatically called during initialization of any
            IO object, so it is not necessary to call it directly.

        """
        if not os.path.isfile(self.data_path):
            _tempdataset = netCDF4.Dataset(
                self.data_path, "w", format="NETCDF4")
            _tempdataset.close()

        _dataset = xr.open_dataset(self.data_path)
        self.dataset = _dataset.set_coords(['time', 'y', 'x'])

    def read(self, var):
        """Read variable from file and into memory.

        Converts `variables` in netCDF file to `xarray` objects for coersion
        into a :obj:`~deltametrics.cube.Cube` instance.

        Parameters
        ----------
        var : `str`
            Which variable to load from the file.
        """
        try:
            _arr = self.dataset[var]
        except ValueError as e:
            raise e
        self._in_memory_data[var] = _arr.load()

    def write(self):
        """Write data to file.

        Take a :obj:`~deltametrics.cube.Cube` and write it to file.

        .. warning::
            Not Implemented.

        """
        raise NotImplementedError

    def __getitem__(self, var):
        if var in self._in_memory_data.keys():
            return self._in_memory_data[var]
        else:
            return self.dataset.variables[var]

    @property
    def keys(self):
        """Variable names in file.
        """
        return [var for var in self.dataset.variables]


class HDFIO(BaseIO):
    """Utility for consistent IO with HDF5 files.

    This module wraps calls to the hdf5 python module in a consistent API,
    so the user can work seamlessly with either HDF5 files or netCDF4 files.
    The public methods of this class are consistent with
    :obj:`~deltametrics.utils.NetCDFIO`.
    """

    def __init__(self, data_path):
        """Initialize the HDF5_IO handler

        parameters
        """
        raise NotImplementedError
