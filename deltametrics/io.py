
import abc
import os
import sys
from warnings import warn

import numpy as np
import xarray as xr
import netCDF4


class BaseIO(abc.ABC):
    """BaseIO object other file format wrappers inheririt from.

    .. note::
        This is an abstract class and cannot be instantiated directly. If you
        wish to subclass to create a new IO format, you must implement the
        methods ``connect``, ``read``, ``write``, and ``keys``.
    """

    def __init__(self, data_path, type, write):
        """Initialize the base IO.
        """
        self.data_path = data_path
        self.type = type
        self.write = write

        self.connect()

        self.get_known_coords()
        self.get_known_variables()

    @property
    def data_path(self):
        """`str` : Path to data file.

        Parameters
        ----------
        data_path : str
            path to data file for IO operations.

        Notes
        -----
        The setter method validates the path, and returns a
        ``FileNotFoundError`` if the file is not found.
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
    def get_known_variables(self):
        """Should create list of known variables.

        This function needs to populate `self.known_variables`.
        """
        return

    @abc.abstractmethod
    def get_known_coords(self):
        """A list of known coordinates.

        This function needs to populate `self.known_coords`.
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
    """Utility for consistent IO with netCDF4 files.

    This module wraps calls to the netCDF4 python module in a consistent API,
    so the user can work seamlessly with either netCDF4 files or HDF5 files.
    The public methods of this class are consistent with
    :obj:`~deltametrics.utils.HDFIO`.

    Note that the netCDF4, netCDF4-classic, and HDF5 file standards are very
    similar and (almost) interchangable. This means that the same data loader
    can be used to handle these files. We use the `xarray` data reader which
    supports the netCDF4/HDF5 file-format.

    Older file formats such as netCDF3 or HDF4 are unsupported. For more
    information about the netCDF4 format, visit the netCDF4
    `docs <https://www.unidata.ucar.edu/software/netcdf/docs/faq.html>`_.
    """

    def __init__(self, data_path, type, write=False):
        """Initialize the NetCDFIO handler.

        Initialize a connection to a NetCDF file.

        Parameters
        ----------
        data_path : `str`
            Path to file to read or write to.

        type : `str`
            Stores the type of output file loaded, either a netCDF4 file,
            'netcdf' or an HDF5 file, 'hdf5'.

        write : `bool`, optional
            Whether to allow writing to an existing file. Set to False by
            default, if a file already exists at ``data_path``, writing is
            disabled, unless ``write`` is set to True.
        """

        super().__init__(data_path=data_path, type=type, write=write)

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

        try:
            _dataset = xr.open_dataset(self.data_path)

            if 'time' and 'y' and 'x' in _dataset.variables:
                self.dataset = _dataset.set_coords(['time', 'y', 'x'])
            else:
                warn('Dimensions "time", "y", and "x" not provided in the \
                      given data file.', UserWarning)

        except Exception:
            raise TypeError('File format out of scope for DeltaMetrics')

    def get_known_variables(self):
        """List known variables.

        These variables are pulled from the loaded dataset.
        """
        self.known_variables = list(self.dataset.variables)

    def get_known_coords(self):
        """List known coordinates.

        These coordinates are pulled from the loaded dataset.
        """
        self.known_coords = list(self.dataset.coords)

    def read(self, var):
        """Read variable from file and into memory.

        Converts `variables` in data file to `xarray` objects for coersion
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
