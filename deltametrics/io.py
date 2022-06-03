
import abc
import os
import copy
from warnings import warn

import xarray as xr
import numpy as np
import netCDF4


class BaseIO(abc.ABC):
    """BaseIO object other file format wrappers inheririt from.

    .. note::

        This is an abstract class and cannot be instantiated directly. If you
        wish to subclass to create a new IO format, you must implement
        several methods.

        To create an IO format for data already loaded into memory, you can
        subclass `BaseIO` directly, and you just need to implement the
        `__getitem__` method and `keys` attribute.

        To create an IO format that reads data from disk, you should subclass
        `FileIO`, and implement the required methods `__getitem__`, `connect`,
        `read`, and `write`, and the  `keys` attribute.
    """

    def __init__(self, io_type):
        """Initialize the base IO.
        """
        self.io_type = io_type

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


class FileIO(BaseIO):
    """Base class for File input output datasets.

    This class should be the basis for subclasses that read data directly from
    a file or folder.

    To create an IO format that reads data from disk, you should subclass
    `FileIO`, and implement the required methods `__getitem__`, `connect`,
    `read`, and `write`, and the  `keys` attribute.
    """

    def __init__(self, data_path, io_type, write=False):
        """Initialize a file IO handler.

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
        self.data_path = data_path
        self.io_type = io_type
        self.write = write

        self.connect()

        self.get_known_coords()
        self.get_known_variables()

        super().__init__(io_type=io_type)

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

        If no file is required, this function should simply pass.
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


class NetCDFIO(FileIO):
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

    def __init__(self, data_path, io_type, write=False):
        """Initialize the NetCDFIO handler.

        Initialize a connection to a NetCDF file.

        Parameters
        ----------
        data_path : `str`
            Path to file to read or write to.

        io_type : `str`
            Stores the type of output file loaded, either a netCDF4 file,
            'netcdf' or an HDF5 file, 'hdf5'.

        write : `bool`, optional
            Whether to allow writing to an existing file. Set to False by
            default, if a file already exists at ``data_path``, writing is
            disabled, unless ``write`` is set to True.
        """

        super().__init__(data_path=data_path, io_type=io_type, write=write)

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

        _ext = os.path.splitext(self.data_path)[-1]
        if _ext == '.nc':
            _engine = 'netcdf4'
        elif _ext == '.hdf5':
            _engine = 'h5netcdf'
        else:
            TypeError('File format is not supported '
                      'by DeltaMetrics: {0}'.format(_ext))

        try:
            # open the dataset
            _dataset = xr.open_dataset(self.data_path, engine=_engine)
        except Exception as e:
            raise TypeError(
                f'File format out of scope for DeltaMetrics: {e}')

        # try to find if coordinates have been preconfigured
        _coords_list = list(_dataset.coords)
        if len(_coords_list) == 3:
            # the coordinates are preconfigured
            self.dataset = _dataset.set_coords(_coords_list)
            self.coords = list(self.dataset.coords)
            self.dims = copy.deepcopy(self.coords)
        elif set(['total_time', 'length', 'width']).issubset(set(_dataset.dims.keys())):
            # the coordinates are not set, but there are matching arrays
            # this is a legacy option, so issue a warning here
            self.dataset = _dataset.set_coords(['x', 'y', 'time'])
            self.dims = ['time', 'length', 'width']
            self.coords = ['total_time', 'x', 'y']
            warn('Coordinates for "time", and ("y", "x") were found as '
                 'variables in the underlying data file, '
                 'but are not specified as coordinates in the undelying '
                 'data file. Please reformat the data file for use '
                 'with DeltaMetrics. This warning may be replaced '
                 'with an Error in a future version.', UserWarning)
        else:
            # coordinates were not found and are not being set
            raise NotImplementedError(
                'Underlying NetCDF datasets without any specified coordinates '
                'are not supported. See source for additional notes about '
                'how to implement this feature.')
            # DEVELOPER NOTE: it may be possible to support a netcdf file that
            # does not have specified coordinates, but we need a test case to
            # make it work. It may work to just pass everything along to the
            # cube, and let xarray automatically handle the naming of
            # coordinates, but I have not tested this.

            # self.dataset = _dataset.set_coords([])
            # self.dims = []
            # warn('Coordinates for "time", and set("x", "y") not provided in the \
            #       given data file.', UserWarning)

        try:
            _meta = xr.open_dataset(self.data_path, group='meta',
                                    engine=_engine)
            self.meta = _meta
        except OSError:
            warn('No associated metadata was found in the given data file.',
                 UserWarning)
            self.meta = None

    def get_known_variables(self):
        """List known variables.

        These variables are pulled from the loaded dataset.
        """
        _vars = list(self.dataset.variables)
        _coords = list(self.dataset.coords)
        if ('strata_age' in _vars) or ('strata_depth' in _vars):
            _coords += ['strata_age', 'strata_depth']
        self.known_variables = [item for item in _vars if item not in _coords]

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
        except KeyError as e:
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
            return self.dataset[var]

    @property
    def keys(self):
        """Variable names in file.
        """
        return [var for var in self.dataset.variables]


class DictionaryIO(BaseIO):
    """Utility for consistent IO with a dictionary as input.

    This module wraps calls to an underyling data dictionary, so that any
    arbitrary data can be used as a cube dataset.
    """

    def __init__(self, data_dictionary, dimensions=None):
        """Initialize the dictionary handler.

        Parameters
        ----------
        data_dictionary : `dict`
            The dictionary, containing `np.ndarray` or `xr.DataArray` data for
            each variable.
        """

        super().__init__(io_type='dictionary')

        self.dataset = data_dictionary
        self._in_memory_data = self.dataset

        self.get_known_variables()
        self.get_known_coords(dimensions)

    def get_known_variables(self):
        """List known variables.

        These variables are pulled from the loaded dataset.
        """
        _vars = self.dataset.keys()
        self.known_variables = [item for item in _vars]

    def get_known_coords(self, dimensions):
        """List known coordinates.

        These coordinates must be supplied during instantiation.
        """
        # start with a grab of underlying data
        under = list(self.dataset.values())[0]
        under_shp = under.shape

        # if the underlying data variables are xarray,
        #   then we ignore any of the other argument passed
        if isinstance(under, xr.core.dataarray.DataArray):
            # get the coordinates and dimensions from the data
            self.dims = under.dims
            self.coords = [under.coords[dim].data for dim in self.dims]
            self.dimensions = dict(zip(self.dims, self.coords))
        # otherwise, check for the arguments passed
        elif not (dimensions is None):
            # if dimensions was passed, it must be a dictionary
            if not isinstance(dimensions, dict):
                raise TypeError(
                    'Input type for `dimensions` must be '
                    '`dict` but was {0}'.format(type(dimensions)))
            # there should be exactly 3 keys
            if not (len(dimensions.keys()) == 3):
                raise ValueError(
                    '`dimensions` must contain three dimensions!')
            # use the dimensions keys as dims and the vals as coords
            #   note, we check the size against the underlying a we go
            for i, (k, v) in enumerate(dimensions.items()):
                if not (len(dimensions[k]) == under_shp[i]):
                    raise ValueError(
                        'Shape of `dimensions` at position {0} was {1}, '
                        'which does not match the variables dimensions '
                        '{2}.'.format(i, len(dimensions[k]), under_shp))
            # make the assignment
            self.dims = list(dimensions.keys())
            self.coords = list(dimensions.values())
            self.dimensions = dimensions
        # otherwise, fill with np.arange(shape)
        else:
            self.dims = ['dim0', 'dim1', 'dim2']
            coords = []
            for i in range(3):
                coords.append(np.arange(under_shp[i]))
            self.coords = coords

        self.known_coords = self.dims
        self.dimensions = dict(zip(self.dims, self.coords))

    def connect(self, *args, **kwargs):
        """Connect to the data file.

        .. warning::
            Not Implemented.

        """
        raise NotImplementedError

    def read(self, *args, **kwargs):
        """Read variable from file and into memory.

        .. warning::
            Not Implemented. Data is always in memory.

        """
        raise NotImplementedError

    def write(self):
        """Write data to file.

        Take a :obj:`~deltametrics.cube.Cube` and write it to file.

        .. warning::
            Not Implemented.

        """
        raise NotImplementedError

    def __getitem__(self, var):
        """Get item reimplemented for dictionaires.

        Returns the variables exactly as they are: either a numpy ndarray or
        xarray.
        """
        if var in self.dataset.keys():
            return self.dataset[var]
        elif var in self.known_coords:
            return self.dimensions[var]
        else:
            raise ValueError(
                'No variable named {0} found.'.format(var))

    @property
    def keys(self):
        """Variable names in file.
        """
        return [var for var in self.dataset.variables]
