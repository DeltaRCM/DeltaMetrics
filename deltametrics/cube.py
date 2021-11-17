import os
import copy
import abc

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from . import io
from . import plan
from . import section
from . import strat
from . import plot


class BaseCube(abc.ABC):
    """Base cube object.

    Cube objects contain t-x-y or z-x-y information.

    This base class should not be used directly, but is subclassed below,
    providing convenient obejcts for maniplating common data types.

    .. note::
        `Cube` does not load any data into memory by default. This means that
        slicing is handled "behind the scenes" by an :doc:`I/O file handler
        </reference/io/index>`. Optionally, you can load files into memory for
        (sometimes) faster operations.
        See the :meth:`read` for more information.

    """

    def __init__(self, data, read=[], varset=None, coordinates={}):
        """Initialize the BaseCube.

        Parameters
        ----------
        data : :obj:`str`, :obj:`dict`
            If data is type `str`, the string points to a NetCDF or HDF5 file
            that can be read. Typically this is used to directly import files
            output from the pyDeltaRCM model. Alternatively, pass a
            :obj:`dict` with keys indicating variable names, and values with
            corresponding t-x-y `ndarray` of data.

        read : :obj:`bool`, optional
            Which variables to read from dataset into memory. Special option
            for ``read=True`` to read all available variables into memory.

        varset : :class:`~deltametrics.plot.VariableSet`, optional
            Pass a `~deltametrics.plot.VariableSet` instance if you wish
            to style this cube similarly to another cube.
        """
        if type(data) is str:
            # handle a path to netCDF file
            self._data_path = data
            self._connect_to_file(data_path=data)
            self._read_meta_from_file(coordinates)
        elif type(data) is dict:
            # handle a dict, arrays set up already, make an io class to wrap it
            self._data_path = None
            raise NotImplementedError
        elif isinstance(data, DataCube):
            # handle initializing one cube type from another
            self._data_path = data.data_path
            self._dataio = data._dataio
            self._read_meta_from_file(coordinates)
        else:
            raise TypeError('Invalid type for "data": %s' % type(data))

        self._plan_set = {}
        self._section_set = {}

        if varset:
            self.varset = varset
        else:
            self.varset = plot.VariableSet()

    @abc.abstractmethod
    def __getitem__(self, var):
        """Return the variable.

        Overload slicing operations for io to return correct var. Must be
        implemented by subclasses.
        """
        ...

    def _connect_to_file(self, data_path):
        """Connect to file.

        This method is used internally to send the ``data_path`` to the
        correct IO handler.
        """
        _, ext = os.path.splitext(data_path)
        if ext == '.nc':
            self._dataio = io.NetCDFIO(data_path, 'netcdf')
        elif ext == '.hdf5':
            self._dataio = io.NetCDFIO(data_path, 'hdf5')
        else:
            raise ValueError(
                'Invalid file extension for "data_path": %s' % data_path)

    def _read_meta_from_file(self, coordinates):
        """Read metadata information from variables in file.

        This method is used internally to gather some useful info for
        navigating the variable trees in the stored files.

        Parameters
        ----------
        coordinates : :obj:`dict`

            A dictionary describing *substitutions* to make for coordinates in
            the underlying dataset with coordinates in DeltaMetrics.
            Dictionary may be empty (default), in which case we connect the
            `x` coordinate in DeltaMetrics to the `x` coordinate in the
            underlying data file, and similarly for `y`. If the underlying
            data file uses a different convention, for example `x` means
            downstream rather than left-right, you can specify this by
            passing a `key-value` pair describing the `value` in the
            underlying data file that corresponds to the DeltaMetrics `key`.
        """
        default_coordinates = {'x': 'x', 'y': 'y'}
        default_coordinates.update(coordinates)
        self.coordinates = copy.deepcopy(default_coordinates)
        self._coords = self._dataio.known_coords
        self._variables = self._dataio.known_variables

        # process the dimensions into attributes
        if len(self._dataio.dims) > 0:
            d0, d1, d2 = self._dataio.dims
        else:
            # try taking a slice of the dataset to get the coordinates
            d0, d1, d2 = self.dataio[self._dataio.known_variables[0]].dims
        self._dim0_idx = self._dataio.dataset[d0]

        # if x is 2-D then we assume x and y are mesh grid values
        if np.ndim(self._dataio[d1]) == 2:
            self._dim1_idx = self._dataio.dataset[d1][:, 0].squeeze()
            self._dim2_idx = self._dataio.dataset[d2][0, :].squeeze()
        # if x is 1-D we do mesh-gridding
        elif np.ndim(self._dataio[d1]) == 1:
            self._dim1_idx = self._dataio.dataset[d1]
            self._dim2_idx = self._dataio.dataset[d2]

        # assign values to dimensions of cube
        self._dim0_coords = self._t = self._dim0_idx
        self._dim1_coords = self._dim1_idx
        self._dim2_coords = self._dim2_idx

        # DEVELOPER NOTE: can we remvoe the _dimX_idx altogether and just use
        # the _dimX_coords arrays?

    def read(self, variables):
        """Read variable into memory.

        Parameters
        ----------
        variables : :obj:`list` of :obj:`str`, :obj:`str`
            Which variables to read into memory.
        """
        if variables is True:  # special case, read all variables
            variables = self.variables
        elif type(variables) is str:
            variables = [variables]
        else:
            raise TypeError('Invalid type for "variables": %s ' % variables)

        for var in variables:
            self._dataio.read(var)

    @property
    def meta(self):
        return self._dataio.meta

    @property
    def varset(self):
        """:class:`~deltametrics.plot.VariableSet` : Variable styling for plotting.

        Can be set with :code:`cube.varset = VariableSetInstance` where
        ``VariableSetInstance`` is a valid instance of
        :class:`~deltametrics.plot.VariableSet`.
        """
        return self._varset

    @varset.setter
    def varset(self, var):
        if type(var) is plot.VariableSet:
            self._varset = var
        else:
            raise TypeError('Pass a valid VariableSet instance.')

    @property
    def data_path(self):
        """:obj:`str` : Path connected to for file IO.

        Returns a string if connected to file, or None if cube initialized
        from ``dict``.
        """
        return self._data_path

    @property
    def dataio(self):
        """:obj:`~deltametrics.io.BaseIO` subclass : Data I/O handler.
        """
        return self._dataio

    @property
    def coords(self):
        """`list` : List of coordinate names as strings."""
        return self._coords

    @property
    def variables(self):
        """`list` : List of variable names as strings.
        """
        return self._variables

    @property
    def plan_set(self):
        """`dict` : Set of plan instances.
        """
        return self._plan_set

    @property
    def plans(self):
        """`dict` : Set of plan instances.

        Alias to :meth:`plan_set`.
        """
        return self._plan_set

    def register_plan(self, name, PlanInstance):
        """Register a planform to the cube.
        """
        if not issubclass(type(PlanInstance), plan.BasePlan):
            raise TypeError
        if not type(name) is str:
            raise TypeError
        self._plan_set[name] = PlanInstance

    @property
    def section_set(self):
        """:obj:`dict` : Set of section instances.
        """
        return self._section_set

    @property
    def sections(self):
        """:obj:`dict` : Set of section instances.

        Alias to :meth:`section_set`.
        """
        return self._section_set

    def register_section(self, name, SectionInstance, return_section=False):
        """Register a section to the :meth:`section_set`.

        Connect a section to the cube.

        Parameters
        ----------
        name : :obj:`str`
            The name to register the section.

        SectionInstance : :obj:`~deltametrics.section.BaseSection` subclass instance
            The section instance that will be registered.

        return_section : :obj:`bool`
            Whether to return the section object.

        Notes
        -----

        When the API for instantiation of the different section types is
        settled, we should enable the ability to pass section kwargs to this
        method, and then instantiate the section internally. This avoids the
        user having to specify ``dm.section.StrikeSection(y=5)`` in the
        ``register_Section()`` call, and instead can do something like
        ``rcm8cube.register_section('trial', trace='strike', y=5)``.
        """

        if not issubclass(type(SectionInstance), section.BaseSection):
            raise TypeError
        if not type(name) is str:
            raise TypeError
        SectionInstance.connect(self, name=name)  # attach cube
        self._section_set[name] = SectionInstance
        if return_section:
            return self._section_set[name]

    @property
    def dim0_coords(self):
        """Coordinates along the first dimension of `cube`.
        """
        return self.z

    @property
    def dim1_coords(self):
        """Coordinates along the second dimension of `cube`.
        """
        return self._dim1_coords

    @property
    def dim2_coords(self):
        """Coordinates along the third dimension of `cube`.
        """
        return self._dim2_coords

    @property
    @abc.abstractmethod
    def z(self):
        """Vertical coordinate."""
        ...

    @property
    @abc.abstractmethod
    def Z(self):
        """Vertical mesh."""
        ...

    @property
    def H(self):
        """Number of elements, vertical (height) coordinate."""
        return self._H

    @property
    def L(self):
        """Number of elements, length coordinate."""
        return self._L

    @property
    def W(self):
        """Number of elements, width coordinate."""
        return self._W

    @property
    def shape(self):
        """Number of elements in data (HxLxW)."""
        return (self.H, self.L, self.W)

    def export_frozen_variable(self, var, return_cube=False):
        """Export a cube with frozen values.

        Creates a `H x L x W` `ndarray` with values from variable `var` placed
        into the array. This method is particularly useful for inputs to
        operation that will repeatedly utilize the underlying data in
        computations. Access to underlying data is comparatively slow to data
        loaded in memory, because the `Cube` utilities are configured to read
        data off-disk as needed.
        """
        if return_cube:
            raise NotImplementedError
        else:
            return self[var].load()

    def show_cube(self, var, t=-1, x=-1, y=-1, ax=None):
        """Show the cube in a 3d axis.

        3d visualization via `pyvista`.

        .. warning::
            Implementation is crude and should be revisited.
        """
        try:
            import pyvista as pv
        except ImportError:
            ImportError('3d plotting dependency, pyvista, was not found.')

        _grid = pv.wrap(self[var].data.values)
        _grid.plot()

    def show_plan(self, var, t=-1, ax=None, title=None, ticks=False,
                  colorbar_label=False):
        """Show planform image.

        .. warning::
            NEEDS TO BE PORTED OVER TO WRAP THE .show() METHOD OF PLAN!
        """

        _plan = self[var][t]  # REPLACE WITH OBJECT RETURNED FROM PLAN

        # get the extent as arbitrary dimensions
        d0, d1 = _plan.dims
        d0_arr, d1_arr = _plan[d0], _plan[d1]
        _extent = [d1_arr[0],                  # dim1, 0
                   d1_arr[-1] + d1_arr[1],     # dim1, end + dx
                   d0_arr[-1] + d0_arr[1],     # dim0, end + dx
                   d0_arr[0]]                  # dim0, 0

        if not ax:
            ax = plt.gca()

        im = ax.imshow(_plan,
                       cmap=self.varset[var].cmap,
                       norm=self.varset[var].norm,
                       vmin=self.varset[var].vmin,
                       vmax=self.varset[var].vmax,
                       extent=_extent)
        cb = plot.append_colorbar(im, ax)
        if colorbar_label:
            _colorbar_label = \
                self.varset[var].label if (colorbar_label is True) \
                else str(colorbar_label)  # use custom if passed
            cb.ax.set_ylabel(_colorbar_label, rotation=-90, va="bottom")

        if not ticks:
            ax.set_xticks([], minor=[])
            ax.set_yticks([], minor=[])
        if title:
            ax.set_title(str(title))

    def show_section(self, *args, **kwargs):
        """Show a section.

        Can be called by name if section is already registered, or pass a
        fresh section instance and it will be connected.

        Wraps the Section's :meth:`~deltametrics.section.BaseSection.show`
        method.
        """

        # parse arguments
        if len(args) == 0:
            raise ValueError
        elif len(args) == 1:
            SectionInstance = args[0]
            SectionAttribute = None
        elif len(args) == 2:
            SectionInstance = args[0]
            SectionAttribute = args[1]
        else:
            raise ValueError('Too many arguments.')

        # call `show()` from string or by instance
        if type(SectionInstance) is str:
            self.sections[SectionInstance].show(SectionAttribute, **kwargs)
        else:
            if not issubclass(type(SectionInstance), section.BaseSection):
                raise TypeError('You must pass a Section instance, '
                                'or a string matching the name of a '
                                'section registered to the cube.')
            SectionInstance.show(**kwargs)


class DataCube(BaseCube):
    """DataCube object.

    DataCube contains t-x-y information. It may have any
    number of attached attributes (grain size, mud frac, elevation).
    """

    def __init__(self, data, read=[], varset=None, stratigraphy_from=None,
                 coordinates={}):
        """Initialize the BaseCube.

        Parameters
        ----------
        data : :obj:`str`, :obj:`dict`
            If data is type `str`, the string points to a NetCDF or HDF5 file
            that can be read. Typically this is used to directly import files
            output from the pyDeltaRCM model. Alternatively, pass a
            :obj:`dict` with keys indicating variable names, and values with
            corresponding t-x-y `ndarray` of data.

        read : :obj:`bool`, optional
            Which variables to read from dataset into memory. Special option
            for ``read=True`` to read all available variables into memory.

        varset : :class:`~deltametrics.plot.VariableSet`, optional
            Pass a `~deltametrics.plot.VariableSet` instance if you wish
            to style this cube similarly to another cube. If no argument is
            supplied, a new default VariableSet instance is created.

        stratigraphy_from : :obj:`str`, optional
            Pass a string that matches a variable name in the dataset to
            compute preservation and stratigraphy using that variable as
            elevation data. Typically, this is ``'eta'`` in pyDeltaRCM model
            outputs. Stratigraphy can be computed on an existing data cube
            with the :meth:`~deltametrics.cube.DataCube.stratigraphy_from`
            method.
        """
        super().__init__(data, read, varset, coordinates)

        # set up the grid for time
        _, self._T, _ = np.meshgrid(
            self.dim1_coords, self.dim0_coords, self.dim2_coords)

        # get shape from a variable that is not x, y, or time
        i = 0
        while i < len(self.variables):
            if self.variables[i] == 'x' or self.variables[i] == 'y' \
               or self.variables[i] == 'time':
                i += 1
            else:
                _var = self.variables[i]
                i = len(self.variables)

        self._H, self._L, self._W = self[_var].data.shape

        # set up dimension and coordinate fields for when slices of the cube
        # are made
        self._view_dimensions = self._dataio.dims
        self._view_coordinates = copy.deepcopy({
            self._view_dimensions[0]: self.dim0_coords,
            self._view_dimensions[1]: self.dim1_coords,
            self._view_dimensions[2]: self.dim2_coords})

        self._knows_stratigraphy = False

        if stratigraphy_from:
            self.stratigraphy_from(variable=stratigraphy_from)

    def __getitem__(self, var):
        """Return the variable.

        Overload slicing operations for io to return a
        :obj:`~deltametrics.cube.CubeVariable` instance when slicing.

        Parameters
        ----------
        var : :obj:`str`
            Which variable to slice.

        Returns
        -------
        CubeVariable : `~deltametrics.cube.CubeVariable`
            The instantiated CubeVariable.
        """
        if var in self._coords:
            # ensure coords can be called by cube[var]
            if var == 'time':  # special case for time
                _t = np.expand_dims(self.dataio.dataset['time'].values,
                                    axis=(1, 2))
                _xrt = xr.DataArray(
                    np.tile(_t, (1, *self.shape[1:])),
                    coords=self._view_coordinates,
                    dims=self._view_dimensions)
                _obj = _xrt
            else:
                _obj = self._dataio.dataset[var]
            return _obj

        elif var in self._variables:
            _obj = self._dataio.dataset[var]
            return _obj

        else:
            raise AttributeError('No variable of {cube} named {var}'.format(
                                 cube=str(self), var=var))

    def stratigraphy_from(self, variable='eta', style='mesh', **kwargs):
        """Compute stratigraphy attributes.

        Parameters
        ----------
        variable : :obj:`str`, optional
            Which variable to use as elevation data for computing
            preservation. If no value is given for this parameter, we try to
            find a variable `eta` and use that for elevation data if it
            exists.

        style : :obj:`str`, optional
            Which style of stratigraphy to compute, options are :obj:`'mesh'
            <deltametrics.strat.MeshStratigraphyAttributes>` or :obj:`'boxy'
            <deltametrics.strat.BoxyStratigraphyAttributes>`. Additional
            keyword arguments are passed to stratigraphy attribute
            initializers.
        """
        if style == 'mesh':
            self.strat_attr = \
                strat.MeshStratigraphyAttributes(elev=self[variable],
                                                 **kwargs)
        elif style == 'boxy':
            self.strat_attr = \
                strat.BoxyStratigraphyAttributes(elev=self[variable],
                                                 **kwargs)
        else:
            raise ValueError('Bad "style" argument supplied: %s' % str(style))
        self._knows_stratigraphy = True

    @property
    def z(self):
        """Vertical coordinate."""
        return self.t

    @property
    def Z(self):
        """Vertical mesh."""
        return self.T

    @property
    def t(self):
        """time coordinate."""
        return self._t

    @property
    def T(self):
        """Vertical mesh."""
        return self._T


class StratigraphyCube(BaseCube):
    """StratigraphyCube object.

    A cube of precomputed stratigraphy. This is a z-x-y matrix defining
    variables at specific voxel locations.

    This is a special case of a cube.

    """
    @staticmethod
    def from_DataCube(DataCubeInstance, stratigraphy_from='eta',
                      dz=None, z=None, nz=None):
        """Create from a DataCube.

        Examples
        --------
        Create a stratigraphy cube from the example ``rcm8cube``:

        >>> rcm8cube = dm.sample_data.rcm8()
        >>> sc8cube = dm.cube.StratigraphyCube.from_DataCube(rcm8cube, dz=0.05)

        Parameters
        ----------
        DataCubeInstance : :obj:`DataCube`
            The `DataCube` instance to derive from for the new
            StratigraphyCube.

        stratigraphy_from : :obj:`str`, optional
            A string that matches a variable name in the dataset to
            compute preservation and stratigraphy using that variable as
            elevation data. Typically, this is ``'eta'`` in pyDeltaRCM model
            outputs.

        dz : :obj:`float`, optional
            Vertical interval (i.e., resolution) for stratigraphy in new
            StratigraphyCube.

        Returns
        -------
        StratigraphyCubeInstance : :obj:`StratigraphyCube`
            The new `StratigraphyCube` instance.
        """
        return StratigraphyCube(DataCubeInstance,
                                stratigraphy_from=stratigraphy_from,
                                coordinates=DataCubeInstance.coordinates,
                                dz=dz, z=z, nz=nz)

    def __init__(self, data, read=[], varset=None,
                 stratigraphy_from=None, coordinates={},
                 dz=None, z=None, nz=None):
        """Initialize the StratigraphicCube.

        Any instantiation pathway must configure :obj:`z`, :obj:`H`, :obj:`L`,
        :obj:`W`, and :obj:`strata`.

        Parameters
        ----------
        data : :obj:`str`, :obj:`dict`
            If data is type `str`, the string points to a NetCDF or HDF5 file
            that can be read. Typically this is used to directly import files
            output from the pyDeltaRCM model. Alternatively, pass a
            :obj:`dict` with keys indicating variable names, and values with
            corresponding t-x-y `ndarray` of data.

        read : :obj:`bool`, optional
            Which variables to read from dataset into memory. Special option
            for ``read=True`` to read all available variables into memory.

        varset : :class:`~deltametrics.plot.VariableSet`, optional
            Pass a `~deltametrics.plot.VariableSet` instance if you wish
            to style this cube similarly to another cube. If no argument is
            supplied, a new default VariableSet instance is created.
        """
        super().__init__(data, read, varset, coordinates)
        if isinstance(data, str):
            raise NotImplementedError('Precomputed NetCDF?')
        elif isinstance(data, np.ndarray):
            raise NotImplementedError('Precomputed numpy array?')
        elif isinstance(data, DataCube):
            # i.e., creating from a DataCube
            _elev = copy.deepcopy(data[stratigraphy_from])

            # set up coordinates of the array
            _z = strat._determine_strat_coordinates(
                _elev.data, dz=dz, z=z, nz=nz)
            self._z = xr.DataArray(_z, name='z', dims=['z'], coords={'z': _z})
            self._H = len(self.z)
            self._L, self._W = _elev.shape[1:]
            self._Z = np.tile(self.z, (self.W, self.L, 1)).T

            _out = strat.compute_boxy_stratigraphy_coordinates(
                _elev, z=_z, return_strata=True)
            self.strata_coords, self.data_coords, self.strata = _out
        else:
            raise TypeError('No other input types implemented yet.')

        self._view_dimensions = ['z', *data._dataio.dims[1:]]
        self._view_coordinates = copy.deepcopy({
            self._view_dimensions[0]: self._z,
            self._view_dimensions[1]: self.dim1_coords,
            self._view_dimensions[2]: self.dim2_coords})

    def __getitem__(self, var):
        """Return the variable.

        Overload slicing operations for io to return a
        :obj:`~deltametrics.cube.CubeVariable` instance when slicing, where
        the data have been placed into stratigraphic position.

        Parameters
        ----------
        var : :obj:`str`
            Which variable to slice.

        Returns
        -------
        CubeVariable : `~deltametrics.cube.CubeVariable`
            The instantiated CubeVariable.
        """
        if var == 'time':
            # a special attribute we add, which matches eta.shape
            _t = np.expand_dims(self.dataio['time'], axis=(1, 2))
            _arr = np.full(self.shape, np.nan)
            _var = np.tile(_t, (1, *self.shape[1:]))
        elif var in self._variables:
            _arr = np.full(self.shape, np.nan)
            # _var = np.array(self.dataio[var], copy=True)
            _var = self.dataio[var]
        else:
            raise AttributeError('No variable of {cube} named {var}'.format(
                                 cube=str(self), var=var))

        # the following lines apply the data to stratigraphy mapping
        if isinstance(_var, xr.core.dataarray.DataArray):
            _vardata = _var.data
        else:
            _vardata = _var
        _cut = _vardata[self.data_coords[:, 0], self.data_coords[:, 1],
                        self.data_coords[:, 2]]
        _arr[self.strata_coords[:, 0], self.strata_coords[:, 1],
             self.strata_coords[:, 2]] = _cut
        _obj = xr.DataArray(
            _arr,
            coords=self._view_coordinates,
            dims=self._view_dimensions)
        return _obj

    @property
    def strata(self):
        """Strata surfaces.

        Elevation of stratal surfaces, matched to times in :obj:`time`.
        """
        return self._strata

    @strata.setter
    def strata(self, var):
        self._strata = var

    @property
    def z(self):
        return self._z

    @property
    def Z(self):
        """Vertical mesh."""
        return self._Z
