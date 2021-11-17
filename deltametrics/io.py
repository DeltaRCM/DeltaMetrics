
import abc
import os
from warnings import warn

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
        if set(['time', 'x', 'y']).issubset(set(_coords_list)):
            # the coordinates are preconfigured
            self.dataset = _dataset.set_coords(_coords_list)
            self.dims = list(self.dataset.dims)
            self.coords = list(self.dataset.coords)
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
