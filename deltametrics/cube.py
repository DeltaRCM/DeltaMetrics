import os
import copy
import warnings
import time
import abc

import numpy as np
import scipy as sp
import xarray as xr

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from . import io
from . import plan
from . import section
from . import strat
from . import utils
from . import plot


@xr.register_dataarray_accessor("cubevar")
class CubeVariable():
    """Slice of a Cube.

    Slicing an :obj:`~deltametrics.cube.Cube` returns an object of this type.
    The ``CubeVariable`` is essentially a thin wrapper around the numpy
    ``ndarray``, enabling additional iniformation to be augmented, a
    lighter-weight __repr__, and flexibility for development.

    .. warning::
        You probably should not instantiate objects of this type directly.

    Examples
    --------
    .. doctest::

        >>> import deltametrics as dm
        >>> rcm8cube = dm.sample_data.cube.rcm8()

        >>> type(rcm8cube['velocity'])
        <class 'deltametrics.cube.CubeVariable'>

        >>> type(rcm8cube['velocity'].base)
        <class 'numpy.ndarray'>

        >>> rcm8cube['velocity'].variable
        'velocity'
    """

    def __init__(self, xarray_obj):
        """Initialize the ``CubeVariable`` object."""
        self.data = xarray_obj

    def initialize(self, **kwargs):
        """Initialize with **kwargs."""
        self.shape = self.data.shape
        self.ndim = len(self.shape)
        variable = kwargs.pop('variable', None)
        self.variable = variable
        coords = kwargs.pop('coords', None)
        if not coords:
            self.t, self.x, self.y = [np.arange(itm) for itm in self.shape]
        else:
            self.t, self.x, self.y = coords['t'], coords['x'], coords['y']

    def as_frozen(self):
        """Export variable as `ndarray`."""
        return self.data.values


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

    def __init__(self, data, read=[], varset=None):
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
            self._read_meta_from_file()
        elif type(data) is dict:
            # handle a dict, arrays set up already, make an io class to wrap it
            self._data_path = None
            raise NotImplementedError
        elif isinstance(data, DataCube):
            # handle initializing one cube type from another
            self._data_path = data.data_path
            self._dataio = data._dataio
            self._read_meta_from_file()
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

    def _read_meta_from_file(self):
        """Read metadata information from variables in file.

        This method is used internally to gather some useful info for
        navigating the variable trees in the stored files.
        """
        self._variables = self._dataio.keys
        # if x is 2-D then we assume x and y are mesh grid values
        if np.ndim(self._dataio['x']) == 2:
            self._X = self._dataio['x']  # mesh grid of x values of cube
            self._Y = self._dataio['y']  # mesh grid of y values of cube
            self._x = np.copy(self._X[0, :].squeeze())  # array of xval of cube
            self._y = np.copy(self._Y[:, 0].squeeze())  # array of yval of cube
        # if x is 1-D we do mesh-gridding
        elif np.ndim(self._dataio['x']) == 1:
            self._x = self._dataio['x']  # array of xval of cube
            self._y = self._dataio['y']  # array of yval of cube
            self._X, self._Y = np.meshgrid(self._x, self._y)  # mesh grids x&y

    def read(self, variables):
        """Read variable into memory

        Parameters
        ----------
        variables : :obj:`list` of :obj:`str`, :obj:`str`
            Which variables to read into memory.
        """
        if variables is True:  # special case, read all variables
            variables = self._dataio.variables
        elif type(variables) is str:
            variables = [variables]
        else:
            raise TypeError('Invalid type for "variables": %s ' % variables)

        for var in variables:
            self._dataio.read(var)

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

        Returns a string if connected to file, or None if cube initialized from ``dict``.
        """
        return self._data_path

    @property
    def dataio(self):
        """:obj:`~deltametrics.io.BaseIO` subclass : Data I/O handler.
        """
        return self._dataio

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

    def register_section(self, name, SectionInstance):
        """Register a section to the :meth:`section_set`.

        Connect a section to the cube.

        Parameters
        ----------
        name : :obj:`str`
            The name to register the section.

        SectionInstance : :obj:`~deltametrics.section.BaseSection` subclass instance
            The section instance that will be registered.

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
        SectionInstance.connect(self)  # attach cube
        self._section_set[name] = SectionInstance

    @property
    def x(self):
        """x-direction coordinate."""
        return self._x

    @property
    def X(self):
        """x-direction mesh."""
        return self._X

    @property
    def y(self):
        """y-direction coordinate."""
        return self._y

    @property
    def Y(self):
        """y-direction mesh."""
        return self._Y

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
            return self[var].data

    def show_cube(self, var, t=-1, x=-1, y=-1, ax=None):
        """Show the cube in a 3d axis.
        """
        raise NotImplementedError

    def show_plan(self, var, t=-1, ax=None, title=None, ticks=False):
        """Show planform image.

        .. warning::
            NEEDS TO BE PORTED OVER TO WRAP THE .show() METHOD OF PLAN!
        """

        _plan = self[var].data[t]  # REPLACE WITH OBJECT RETURNED FROM PLAN

        if not ax:
            ax = plt.gca()

        im = ax.imshow(_plan,
                       cmap=self.varset[var].cmap,
                       norm=self.varset[var].norm,
                       vmin=self.varset[var].vmin,
                       vmax=self.varset[var].vmax)
        cb = plot.append_colorbar(im, ax)

        if not ticks:
            ax.set_xticks([], minor=[])
            ax.set_yticks([], minor=[])
        if title:
            ax.set_title(str(title))

        ax.set_xlim(self.x[0], self.x[-1])
        ax.set_ylim(self.y[0], self.y[-1])

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

    def __init__(self, data, read=[], varset=None, stratigraphy_from=None):
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
        super().__init__(data, read, varset)

        self._t = np.array(self._dataio['time'], copy=True)
        _, self._T, _ = np.meshgrid(self.y, self.t, self.x)

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
        if var == 'time':
            # a special attribute we add, which matches eta.shape
            _coords = {}
            _coords['t'] = self.T
            _coords['x'] = self.X
            _coords['y'] = self.Y
            _obj = self._dataio.dataset[var].cubevar
            _obj.initialize(variable='time', coords=_coords)
            return _obj
            # return CubeVariable(np.tile(_t, (1, *self.shape[1:])),
            #                     variable='time')
        elif var in self._variables:
            _obj = self._dataio.dataset[var].cubevar
            _obj.initialize(variable=var)
            return _obj
            # return CubeVariable(self.dataio[var], variable=var)
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
                strat.MeshStratigraphyAttributes(elev=self[variable].data,
                                                 **kwargs)
        elif style == 'boxy':
            self.strat_attr = \
                strat.BoxyStratigraphyAttributes(elev=self[variable].data,
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
    def from_DataCube(DataCubeInstance, stratigraphy_from='eta', dz=0.1):
        """Create from a DataCube.

        Examples
        --------
        Create a stratigraphy cube from the example ``rcm8cube``:

        >>> rcm8cube = dm.sample_data.cube.rcm8()
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
                                dz=dz)

    def __init__(self, data, read=[], varset=None,
                 stratigraphy_from=None, dz=None):
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
        super().__init__(data, read, varset)
        if isinstance(data, str):
            raise NotImplementedError('Precomputed NetCDF?')
        elif isinstance(data, np.ndarray):
            raise NotImplementedError('Precomputed numpy array?')
        elif isinstance(data, DataCube):
            # i.e., creating from a DataCube
            _elev = copy.deepcopy(data[stratigraphy_from])

            # set up coordinates of the array
            self._z = strat._determine_strat_coordinates(_elev.data, dz=dz)
            self._H = len(self.z)
            self._L, self._W = _elev.shape[1:]
            self._Z = np.tile(self.z, (self.W, self.L, 1)).T

            _out = strat.compute_boxy_stratigraphy_coordinates(_elev.data,
                                                               z=self.z,
                                                            return_strata=True)
            self.strata_coords, self.data_coords, self.strata = _out
        else:
            raise TypeError('No other input types implemented yet.')

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
            _var = np.array(self.dataio[var], copy=True)
        else:
            raise AttributeError('No variable of {cube} named {var}'.format(
                                 cube=str(self), var=var))

        # the following lines apply the data to stratigraphy mapping
        _cut = _var[self.data_coords[:, 0], self.data_coords[:, 1],
                    self.data_coords[:, 2]]
        _arr[self.strata_coords[:, 0], self.strata_coords[:, 1],
             self.strata_coords[:, 2]] = _cut
        _obj = xr.DataArray(_arr)
        _obj.cubevar.initialize(variable=var)
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
