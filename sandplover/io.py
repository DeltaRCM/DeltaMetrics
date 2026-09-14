import abc
import copy
import os
import warnings

import netCDF4
import numpy as np
import psutil
import xarray as xr


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
        """Initialize the base IO."""
        self.io_type = io_type
        self._aux = None  # default None

        # set of variables in underlying data
        self._underlying_variables = []  # static list to be a populated,
        # set of variables that can be added through registration after instantiation
        self._in_memory_variables = {}
        self._in_memory_data = self._in_memory_variables  # alias for backwards compat

    @abc.abstractmethod
    def __getitem__(self):
        """Should slice the data from underlying data.

        Must be implemented to appropriately slice from file, dict, folder
        structure etc. Must be implemented to appropriately search underlying
        data variables and variables in :attr:`_in_memory_data` during
        slice.
        """
        return

    @abc.abstractmethod
    def _set_aux(self):
        """Should set auxiliary group."""
        return

    def _register_variable(self, name, data):
        """Adds variable to DataIO layer.

        This function is declared private. Assumed to have already been checked
        for shape and type etc.

        Always in memory variable at first. Could implement options to write to
        disk in future releases.
        """
        # self._registered_variables[name] = data
        self._in_memory_variables[name] = data

    @property
    @abc.abstractmethod
    def keys(self):
        """Should link to all key _names_ available in dataio layer."""
        return

    @property
    def known_variables(self):
        """Variables known to the dataio layer

        Includes underlying data variables and registered variables.
        """
        return self._underlying_variables + [*self._in_memory_variables]

    @property
    def aux(self):
        return self._aux

    @property
    def meta(self):
        """alias for backwards compatability"""
        # will be removed in future release.
        return self._aux


class FileIO(BaseIO):
    """Base class for File input output datasets.

    This class should be the basis for subclasses that read data directly from
    a file or folder.

    To create an IO format that reads data from disk, you should subclass
    `FileIO`, and implement the required methods `__getitem__`, `connect`,
    `read`, and `write`, and the  `keys` attribute.
    """

    def __init__(self, data_path, auxdata_path, write=False):
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
        super().__init__(io_type="file")

        self.data_path = data_path
        self.auxdata_path = auxdata_path

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
            raise FileNotFoundError("File not found at supplied path: %s" % var)

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
        """Should read data into memory."""
        return

    @abc.abstractmethod
    def write(self):
        """Should write the data to file.

        Take a :obj:`~sandplover.cube.Cube` and write it to file.
        """
        return


class NetCDFIO(FileIO):
    """Utility for consistent IO with netCDF4 files.

    This module wraps calls to the netCDF4 python module in a consistent API,
    so the user can work seamlessly with either netCDF4 files or HDF5 files.
    The public methods of this class are consistent with
    :obj:`~sandplover.utils.HDFIO`.

    Note that the netCDF4, netCDF4-classic, and HDF5 file standards are very
    similar and (almost) interchangable. This means that the same data loader
    can be used to handle these files. We use the `xarray` data reader which
    supports the netCDF4/HDF5 file-format.

    Older file formats such as netCDF3 or HDF4 are unsupported. For more
    information about the netCDF4 format, visit the netCDF4
    `docs <https://www.unidata.ucar.edu/software/netcdf/docs/faq.html>`_.
    """

    def __init__(self, data_path, auxdata_path=None, engine=None, write=False):
        """Initialize the NetCDFIO handler.

        Initialize a connection to a NetCDF file.

        Parameters
        ----------
        data_path : `str`
            Path to file to read or write to.

        auxdata_path : `str`, optional
            Path to auxilliary data that exist in the file. This is most
            commonly the name of a group within the file. Default is None, and
            no auxilliary data is assigned.

        engine : `str`, optional
            Engine used to open the file with xarray. Default is None, which
            will lead to trying to infer from file extension. If no inference
            can be made, we pass no engine during loading and allow xarray to
            attempt to determine the file type. For a netCDF4 file use 'netcdf4'
            or for an HDF5 file use 'h5netcdf', or any other valid engine for
            xarray.

        write : `bool`, optional
            Whether to allow writing to an existing file. Set to False by
            default, if a file already exists at ``data_path``, writing is
            disabled, unless ``write`` is set to True.
        """
        # set engine used to open the file
        if engine is not None:
            self._engine = engine
        else:
            # attempt to guess
            _, ext = os.path.splitext(data_path)
            if ext == ".nc":
                self._engine = "netcdf4"
            elif ext == ".hdf5":
                self._engine = "h5netcdf"
            else:
                self._engine = None  # let xarray figure it out

        super().__init__(data_path=data_path, auxdata_path=auxdata_path, write=write)

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
            _tempdataset = netCDF4.Dataset(self.data_path, "w", format="NETCDF4")
            _tempdataset.close()

        try:
            # open the dataset
            _dataset = xr.open_datatree(self.data_path, engine=self._engine)
        except Exception as e:
            raise TypeError(
                f"Could not open dataset, raising error: {e}.\n\n"
                f"This may be because the file is corrupted, not recognized, "
                f"or not supported by xarray or sandplover."
            ) from e

        # try to find if coordinates have been preconfigured
        _coords_list = list(_dataset.coords)
        # with warnings.catch_warnings():
        #     # filter warning about Dataset.dims changing return, we use the
        #     # correct use already
        #     warnings.filterwarnings("ignore", category=FutureWarning)
        #     _dims_set = set(_dataset.dims.keys())
        if len(_coords_list) == 3:
            # the coordinates are preconfigured
            self.dataset = _dataset
            self.coords = list(self.dataset.coords)
            self.dims = copy.deepcopy(self.coords)
        elif len(_coords_list) == 4:
            raise NotImplementedError("sandplover does not currently support 4D data.")
            # this is a hard check prohibiting 4d data. To fully support the
            # sandsuet v1.0 spec, we will need to be able to open this type
            # of data file. The different cube types will then have to
            # understand how to use (or disallow) 4D data.
        else:
            # coordinates were not found and are not being set
            raise NotImplementedError(
                "Underlying NetCDF datasets without any specified coordinates "
                "are not supported. See source for additional notes about "
                "how to implement this feature."
            )
            # DEVELOPER NOTE: it may be possible to support a netcdf file that
            # does not have specified coordinates, but we need a test case to
            # make it work. It may work to just pass everything along to the
            # cube, and let xarray automatically handle the naming of
            # coordinates, but I have not tested this.

            # self.dataset = _dataset.set_coords([])
            # self.dims = []
            # warn('Coordinates for "time", and set("x", "y") not provided in the \
            #       given data file.', UserWarning)
        self._set_aux(auxdata_path=self.auxdata_path)

    def _set_aux(self, auxdata_path):
        """Set auxiliary group (declared private).

        Defined as a function so it can also be called by the public
        `Cube.set_aux` method.

        Parameters
        ----------
        auxdata_path : auxiliary data path. See specifications in init docstring.
        """
        # if something was specified for auxdata, set it accordingly
        if not auxdata_path is None:
            self.auxdata_path = auxdata_path
            self._aux = self.dataset[self.auxdata_path]

    def get_known_variables(self):
        """List known variables.

        These variables are pulled from the loaded dataset.
        """
        _vars = list(self.dataset.variables)
        _coords = list(self.dataset.coords)
        if ("strata_age" in _vars) or ("strata_depth" in _vars):
            _coords += ["strata_age", "strata_depth"]
        self._underlying_variables = [item for item in _vars if item not in _coords]

    def get_known_coords(self):
        """List known coordinates.

        These coordinates are pulled from the loaded dataset.
        """
        self.known_coords = list(self.dataset.coords)

    def read(self, var, force=False):
        """Read variable from file and into memory.

        Converts `variables` in data file to `xarray` objects for coersion
        into a :obj:`~sandplover.cube.Cube` instance.

        Parameters
        ----------
        var : `str`
            Which variable to load from the file.

        force : `bool`, optional
            If True, bypass memory safety checks and load the data regardless
            of size. Default is False.

        Warnings
        --------
        If the variable size exceeds 80% of currently available RAM and
        `force=False`, a warning will be issued and the data will NOT be
        loaded into memory. Set `force=True` to override this check.
        """
        try:
            _arr = self.dataset[var]
        except KeyError as e:
            raise e

        # Check memory safety before loading
        if not force:
            # Calculate the memory footprint of the variable
            itemsize = _arr.dtype.itemsize
            total_elements = np.prod(_arr.shape)
            var_size_bytes = itemsize * total_elements

            # Get currently available memory
            available_mem = psutil.virtual_memory().available
            threshold = 0.8 * available_mem

            # Check if variable is too large
            if var_size_bytes > threshold:
                var_size_gb = var_size_bytes / (1024**3)
                available_gb = available_mem / (1024**3)
                threshold_gb = threshold / (1024**3)

                warnings.warn(
                    f"Variable '{var}' is too large to safely load into memory.\n"
                    f"  Variable size: {var_size_gb:.2f} GB\n"
                    f"  Available memory: {available_gb:.2f} GB\n"
                    f"  Safety threshold (80%): {threshold_gb:.2f} GB\n"
                    f"Data was NOT loaded. To override this check and load anyway, "
                    f"call read() with force=True:\n"
                    f"  cube.read('{var}', force=True)",
                    UserWarning,
                    stacklevel=2,
                )
                return  # Exit without loading

        self._in_memory_variables[var] = _arr.load()

    def write(self):
        """Write data to file.

        Take a :obj:`~sandplover.cube.Cube` and write it to file.

        .. warning::
            Not Implemented.

        """
        raise NotImplementedError

    def __getitem__(self, var):
        if var in self._in_memory_variables:
            return self._in_memory_variables[var]
        else:
            return self.dataset[var]

    @property
    def keys(self):
        """Variable names in file."""
        return list(self.dataset.variables)


class DictionaryIO(BaseIO):
    """Utility for consistent IO with a dictionary as input.

    This module wraps calls to an underyling data dictionary, so that any
    arbitrary data can be used as a cube dataset.
    """

    def __init__(self, data_dictionary, auxdata_path=None, dimensions=None):
        """Initialize the DictionaryIO handler.

        Parameters
        ----------
        data_dictionary : `dict`
            Dictionary with `np.ndarray` or `xr.DataArray` arrays containing the
            dataset of interest. All arrays in dict

        auxdata_path : `str`, `dict`, optional
            Path to auxilliary data within the dictionary `data_dictionary`, or
            another dictionary to be treated as auxiliary data. Default is None, and
            no auxilliary data is assigned.

        dimensions : `dict`, optional

            Dimensions of the data in the `data_dictionary` and relevant to
            `aux_datapath`. If any inputs to `data_dictionary` are xarray.DataArray,
            then `dimensions` is ignored, and the dimensions of that `DataArray` are
            applied to all data. Otherwise, provide a dictionary `dimensions` that
            is applied to data in `data_dictionary`. Finally, if `dimensions` is
            None, dimensions are inferred from the first three dimensional variable
            in `data_dictionary`.
        """
        super().__init__(io_type="dictionary")

        self.dataset = data_dictionary
        # in DictionaryIO, overwrite the existing in_memory_variables with the full dataset
        self._in_memory_variables = self.dataset
        self._in_memory_data = self._in_memory_variables  # alias for backwards compat

        # set the auxiliary group
        self._set_aux(auxdata_path)

        self.get_known_variables()
        self.get_known_coords(dimensions)

    def _set_aux(self, auxdata_path):
        """Set auxiliary group (declared private).

        Defined as a function so it can also be called by the public
        `Cube.set_aux` method.

        Parameters
        ----------
        auxdata_path : auxiliary data path. See specifications in init docstring.
        """
        # if something was specified for auxdata, set it accordingly
        if not auxdata_path is None:
            # can be either a string (a dict in a dict) or a separate dict
            if isinstance(auxdata_path, str):
                self.auxdata_path = auxdata_path  # store it
                self._aux = self.dataset[auxdata_path]
            elif isinstance(auxdata_path, dict):
                self.auxdata_path = None  # store None, is dict, no path
                self._aux = auxdata_path
            else:
                raise TypeError(
                    f"Invalid type for DictionaryIO `auxdata_path`. Must be str or dict, but was {type(auxdata_path)}."
                )

            # now verify that aux is dict (requirement for DictionaryIO)
            if not isinstance(self._aux, dict):
                raise TypeError(
                    f"Auxiliary information found at `auxdata_path` was not dict but was {type(self._aux)}"
                )

    def get_known_variables(self):
        """List known variables."""
        _vars = self.dataset.keys()
        self._underlying_variables = list(_vars)

    def get_known_coords(self, dimensions):
        """List known coordinates.

        Priority:
          1) If any value is an xarray.DataArray -> IGNORE `dimensions`
             and use its dims/coords.
          2) Else if `dimensions` provided -> validate against first
             3-D var's shape.
          3) Else -> infer from first 3-D var; if none, error.
        """
        values = list(self.dataset.values())

        # (1) Any xarray.DataArray present? Ignore `dimensions` (legacy behavior).
        xr_arrays = [v for v in values if isinstance(v, xr.DataArray)]
        if xr_arrays:
            under = xr_arrays[0]
            self.dims = list(under.dims)
            self.coords = [under.coords[d].data for d in self.dims]
            self.dimensions = dict(zip(self.dims, self.coords, strict=True))
            self.known_coords = self.dims
            return

        # Helper: find first 3-D ndarray-like to act as reference
        def _first_3d_shape(arrs):
            for v in arrs:
                a = np.asarray(v)
                if a.ndim == 3:
                    return tuple(a.shape)
            return None

        # (2) Dimensions provided
        if dimensions is not None:
            if not isinstance(dimensions, dict):
                raise TypeError(
                    "Input type for `dimensions` must be `dict` but was {}".format(
                        type(dimensions)
                    )
                )
            if len(dimensions) != 3:
                raise ValueError("`dimensions` must contain exactly three dimensions!")

            # Preserve insertion order; ensure 1-D coords
            self.dims = list(dimensions.keys())
            self.coords = [np.asarray(dimensions[k]) for k in self.dims]
            for k, c in zip(self.dims, self.coords):
                if c.ndim != 1:
                    raise ValueError(f"Coordinate '{k}' must be 1-D, got {c.ndim}D.")

            ref_shp = _first_3d_shape(values)
            if ref_shp is not None:
                # Validate like the original (wording kept for tests)
                for i, k in enumerate(self.dims):
                    if len(dimensions[k]) != ref_shp[i]:
                        raise ValueError(
                            "Shape of `dimensions` at position {} was {}, "
                            "which does not match the variables dimensions {}.".format(
                                i, len(dimensions[k]), ref_shp
                            )
                        )

            self.dimensions = dict(zip(self.dims, self.coords, strict=True))
            self.known_coords = self.dims
            return

        # (3) No dimensions: infer from first 3-D var
        ref_shp = _first_3d_shape(values)
        if ref_shp is None:
            raise ValueError(
                "Cannot infer coordinates: supply `dimensions` or include at "
                "least one 3-D variable."
            )
        self.dims = ["dim0", "dim1", "dim2"]
        self.coords = [np.arange(n) for n in ref_shp]
        self.dimensions = dict(zip(self.dims, self.coords, strict=True))
        self.known_coords = self.dims

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

        .. warning::
            Not Implemented.
        """
        raise NotImplementedError

    def __getitem__(self, var):
        """Get item reimplemented for dictionaries."""
        if var in self.dataset:
            return self.dataset[var]
        elif var in self.known_coords:
            return self.dimensions[var]
        else:
            raise ValueError(f"No variable named {var} found.")

    @property
    def keys(self):
        """Variable names in 'file' (dict keys)."""
        return list(self.dataset.keys())
