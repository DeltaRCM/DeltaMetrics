import os
import copy
import abc
import warnings

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

    def __init__(self, data, read=[], varset=None, dimensions=None):
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

        dimensions : `dict`, optional
            A dictionary with names and coordinates for dimensions of the
            cube, if instantiating the cube from data loaded in memory
            in a dictionary.
        """
        if type(data) is str:
            # handle a path to netCDF file
            self._data_path = data
            self._connect_to_file(data_path=data)
            self._read_meta_from_file()
        elif type(data) is dict:
            # handle a dict, arrays set up already, make an io class to wrap it
            self._data_path = None
            self._dataio = io.DictionaryIO(
                data, dimensions=dimensions)
            self._read_meta_from_file()
        elif isinstance(data, DataCube):
            # handle initializing one cube type from another
            self._data_path = data.data_path
            self._dataio = data._dataio
            self._read_meta_from_file()
        else:
            raise TypeError('Invalid type for "data": %s' % type(data))

        self._planform_set = {}
        self._section_set = {}

        if varset:
            self.varset = varset
        else:
            self.varset = plot.VariableSet()

        # some undocumented aliases
        self.plans = self._planform_set
        self.plan_set = self._planform_set

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
        self._coords = self._dataio.known_coords
        self._variables = self._dataio.known_variables

        # process the dimensions into attributes
        if len(self._dataio.dims) > 0:
            d0, d1, d2 = self._dataio.dims
        else:
            # try taking a slice of the dataset to get the coordinates
            d0, d1, d2 = self.dataio[self._dataio.known_variables[0]].dims
        self._dim0_idx = self._dataio[d0]

        # if x is 2-D then we assume x and y are mesh grid values
        if np.ndim(self._dataio[d1]) == 2:
            self._dim1_idx = self._dataio.dataset[d1][:, 0].squeeze()
            self._dim2_idx = self._dataio.dataset[d2][0, :].squeeze()
        # if x is 1-D we do mesh-gridding
        elif np.ndim(self._dataio[d1]) == 1:
            self._dim1_idx = self._dataio[d1]
            self._dim2_idx = self._dataio[d2]
        else:
            raise TypeError(
                'Shape of coordinate array was not 1d or 2d. '
                'Maybe the name was not correctly identified in the dataset, '
                'or the array is misformatted.')

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
    def planform_set(self):
        """:obj:`dict` : Set of planform instances.
        """
        return self._planform_set

    @property
    def planforms(self):
        """`dict` : Set of plan instances.

        Alias to :meth:`plan_set`.
        """
        return self._planform_set

    def register_plan(self, *args, **kwargs):
        """wrapper, might not really need this."""
        return self.register_planform(*args, **kwargs)

    def register_planform(self, name, PlanformInstance, return_planform=False):
        """Register a planform to the :attr:`planform_set`.

        Connect a planform to the cube.

        Parameters
        ----------
        name : :obj:`str`
            The name to register the `Planform`.

        PlanformInstance : :obj:`~deltametrics.planform.BasePlanform` subclass instance
            The planform instance that will be registered.

        return_planform : :obj:`bool`
            Whether to return the planform object.
        """
        if not issubclass(type(PlanformInstance), plan.BasePlanform):
            raise TypeError(
                '`PlanformInstance` was not a `Planform`. '
                'Instead, was: {0}'.format(type(PlanformInstance)))
        if not isinstance(name, str):
            raise TypeError(
                '`name` was not a string. '
                'Instead, was: {0}'.format(type(name)))
        PlanformInstance.connect(self, name=name)  # attach cube
        self._planform_set[name] = PlanformInstance
        if return_planform:
            return self._planform_set[name]

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
        """Register a section to the :attr:`section_set`.

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
        user having to specify ``dm.section.StrikeSection(distance=2000)`` in
        the ``register_section()`` call, and instead can do something like
        ``golf.register_section('trial', trace='strike',
        distance=2000)``.
        """
        if not issubclass(type(SectionInstance), section.BaseSection):
            raise TypeError(
                '`SectionInstance` was not a `Section`. '
                'Instead, was: {0}'.format(type(SectionInstance)))
        if not isinstance(name, str):
            raise TypeError(
                '`name` was not a string. '
                'Instead, was: {0}'.format(type(name)))
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

    @property
    def extent(self):
        """The limits of the dim1 by dim2 plane.

        Useful for plotting.
        """
        _extent = [
            self.dim2_coords[0],                         # dim1, 0
            self.dim2_coords[-1] + self.dim2_coords[1],  # dim1, end + dx
            self.dim1_coords[-1] + self.dim1_coords[1],  # dim0, end + dx
            self.dim1_coords[0]]                         # dim0, 0
        return _extent

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

    def quick_show(self, var, idx=-1, axis=0, **kwargs):
        """Convenient and quick way to show a slice of the cube by `idx` and `axis`.

        .. hint::

            If neither `idx` or `axis` is specified, a planform view of the
            last index is shown.

        Parameters
        ----------
        var : :obj:`str`
            Which variable to show from the underlying dataset.

        idx : :obj:`int`, optional
            Which index along the `axis` to slice data from. Default value is
            ``-1``, the last index along `axis`.

        axis : :obj:`int`, optional
            Which axis of the underlying cube `idx` is specified for. Default
            value is ``0``, the first axis of the cube.

        **kwargs
            Keyword arguments are passed
            to :meth:`~deltametrics.plan.Planform.show` if `axis` is ``0``,
            otherwise passed
            to :meth:`~deltametrics.section.BaseSection.show`.

        Examples
        --------

        .. plot::
            :include-source:

            >>> golfcube = dm.sample_data.golf()
            >>> golfstrat = dm.cube.StratigraphyCube.from_DataCube(
            ...     golfcube, dz=0.1)
            ...
            >>> fig, ax = plt.subplots(2, 1)
            >>> golfcube.quick_show('eta', ax=ax[0])  # a Planform (axis=0)
            >>> golfstrat.quick_show('eta', idx=100, axis=2, ax=ax[1])  # a DipSection
            >>> plt.show()
        """
        if axis == 0:
            # this is a planform slice
            _obj = plan.Planform(self, idx=idx)
        elif axis == 1:
            # this is a Strike section
            _obj = section.StrikeSection(self, distance_idx=idx)
        elif axis == 2:
            # this is a Dip section
            _obj = section.DipSection(self, distance_idx=idx)
        else:
            raise ValueError(
                'Invalid `axis` specified: {0}'.format(axis))

        # use the object to handle the showing
        _obj.show(var, **kwargs)

    def show_cube(self, var, style='mesh', ve=200, ax=None):
        """Show the cube in a 3D axis.

        .. important:: requires `pyvista` package for 3d visualization.

        Parameters
        ----------
        var : :obj:`str`
            Which variable to show from the underlying dataset.

        style : :obj:`str`, optional
            Style to show `cube`. Default is `'mesh'`, which gives a 3D
            volumetric view. Other supported option is `'fence'`, which gives
            a fence diagram with one slice in each cube dimension.

        ve : :obj:`float`
            Vertical exaggeration. Default is ``200``.

        ax : :obj:`~matplotlib.pyplot.Axes` object, optional
            A `matplotlib` `Axes` object to plot the section. Optional; if not
            provided, a call is made to ``plt.gca()`` to get the current (or
            create a new) `Axes` object.

        Examples
        --------

        .. note::

            The following code snippets are not set up to actually make the
            plots in the documentation.

        .. code::

            >>> golfcube = dm.sample_data.golf()
            >>> golfstrat = dm.cube.StratigraphyCube.from_DataCube(
            ...     golfcube, dz=0.1)
            ...
            >>> fig, ax = plt.subplots()
            >>> golfstrat.show_cube('eta', ax=ax)

        .. code::

            >>> golfcube = dm.sample_data.golf()
            ...
            >>> fig, ax = plt.subplots()
            >>> golfcube.show_cube('velocity', style='fence', ax=ax)
        """
        try:
            import pyvista as pv
        except ImportError:
            ImportError(
                '3d plotting dependency, pyvista, was not found.')
        except ModuleNotFoundError:
            ModuleNotFoundError(
                '3d plotting dependency, pyvista, was not found.')
        except Exception as e:
            raise e

        if not ax:
            ax = plt.gca()

        _data = np.array(self[var])
        _data = _data.transpose((2, 1, 0))

        mesh = pv.UniformGrid(_data.shape)
        mesh[var] = _data.ravel(order='F')
        mesh.spacing = (self.dim2_coords[1],
                        self.dim1_coords[1],
                        ve/self.dim1_coords[1])
        mesh.active_scalars_name = var

        p = pv.Plotter()
        p.add_mesh(mesh.outline(), color="k")
        if style == 'mesh':

            threshed = mesh.threshold([-np.inf, np.inf], all_scalars=True)
            p.add_mesh(threshed, cmap=self.varset[var].cmap)

        elif style == 'fence':
            # todo, improve this to manually create the sections so you can
            #   do more than three slices
            slices = mesh.slice_orthogonal()
            p.add_mesh(slices, cmap=self.varset[var].cmap)

        else:
            raise ValueError('Bad value for style: {0}'.format(style))

        p.show()

    def show_plan(self, *args, **kwargs):
        """Deprecated. Use :obj:`quick_show` or :obj:`show_planform`.

        .. warning::

            Provides a legacy option to quickly show a planform, from before
            the `Planform` object was properly implemented. Will be removed
            in a future release.

        Parameters
        ----------
        """
        # legacy method, ported over to show_planform.
        warnings.warn(
            '`show_plan` is a deprecated method, and has been replaced by two '
            'alternatives. To quickly show a planform slice of a cube, you '
            'can use `quick_show()` with a similar API. The `show_planform` '
            'method implements more features, but requires instantiating a '
            '`Planform` object first. Passing arguments to `quick_show`.')
        # pass `t` arg to `idx` for legacy
        if 't' in kwargs.keys():
            idx = kwargs.pop('t')
            kwargs['idx'] = idx

        self.quick_show(*args, **kwargs)

    def show_planform(self, name, variable, **kwargs):
        """Show a registered planform by name and variable.

        Call a registered planform by name and variable.

        Parameters
        ----------
        name : :obj:`str`
            The name of the registered planform.

        variable : :obj:`str`
            The varaible name to show.

        **kwargs
            Keyword arguments passed
            to :meth:`~deltametrics.plan.Planform.show`.
        """
        # call `show()` from string
        if isinstance(name, str):
            self._planform_set[name].show(variable, **kwargs)
        else:
            raise TypeError(
                '`name` was not a string, '
                'was {0}'.format(type(name)))

    def show_section(self, name, variable, **kwargs):
        """Show a registered section by name and variable.

        Call a registered section by name and variable.

        Parameters
        ----------
        name : :obj:`str`
            The name of the registered section.

        variable : :obj:`str`
            The varaible name to show.

        **kwargs
            Keyword arguments passed
            to :meth:`~deltametrics.section.BaseSection.show`.
        """
        # call `show()` from string
        if isinstance(name, str):
            self._section_set[name].show(variable, **kwargs)
        else:
            raise TypeError(
                '`name` was not a string, '
                'was {0}'.format(type(name)))


class DataCube(BaseCube):
    """DataCube object.

    DataCube contains t-x-y information. It may have any
    number of attached attributes (grain size, mud frac, elevation).
    """

    def __init__(self, data, read=[], varset=None, stratigraphy_from=None,
                 dimensions=None):
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

        dimensions : `dict`, optional
            A dictionary with names and coordinates for dimensions of the
            `DataCube`, if instantiating the cube from data loaded in memory
            in a dictionary.
        """
        super().__init__(data, read, varset, dimensions=dimensions)

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

        # set up dimension and coordinate fields for when slices of the cube
        # are made
        self._view_dimensions = self._dataio.dims
        self._view_coordinates = copy.deepcopy({
            self._view_dimensions[0]: self.dim0_coords,
            self._view_dimensions[1]: self.dim1_coords,
            self._view_dimensions[2]: self.dim2_coords})

        # set the shape of the cube
        self._H, self._L, self._W = self[_var].data.shape

        # determine stratigraphy information
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
        if var == 'time':  # special case for time
            # use the name of the first dimension, to enable
            #   unlabeled np.ndarrays and flexible name for time
            dim0_name = self.dataio.dims[0]
            dim0_coord = np.array(self.dataio.dataset[dim0_name])
            _t = np.expand_dims(dim0_coord,
                                axis=(1, 2))
            _xrt = xr.DataArray(
                np.tile(_t, (1, *self.shape[1:])),
                coords=self._view_coordinates,
                dims=self._view_dimensions)
            _obj = _xrt
        elif var in self._coords:
            # ensure coords can be called by cube[var]
            _obj = self._dataio.dataset[var]

        elif var in self._variables:
            _obj = self._dataio.dataset[var]

        else:
            raise AttributeError('No variable of {cube} named {var}'.format(
                                 cube=str(self), var=var))

        # make _obj xarray if it not already
        if isinstance(_obj, np.ndarray):
            _obj = xr.DataArray(
                _obj,
                coords=self._view_coordinates,
                dims=self._view_dimensions)
        return _obj

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

        **kwargs
            Keyword arguments passed to stratigraphy initialization. Can
            include specification for vertical resolution in `Boxy` case,
            see :obj:_determine_strat_coordinates`.
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
                      sigma_dist=None, dz=None, z=None, nz=None):
        """Create from a DataCube.

        Examples
        --------
        Create a stratigraphy cube from the example ``golf``:

        >>> golfcube = dm.sample_data.golf()
        >>> stratcube = dm.cube.StratigraphyCube.from_DataCube(golfcube, dz=0.05)

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

        **kwargs
            Keyword arguments passed to stratigraphy initialization. Can
            include specification for vertical resolution in `Boxy` case,
            see :obj:`~deltametrics.strat._determine_strat_coordinates`,
            as well as information about subsidence,
            see :obj:`~deltametrics.strat._adjust_elevation_by_subsidence`.

        Returns
        -------
        StratigraphyCubeInstance : :obj:`StratigraphyCube`
            The new `StratigraphyCube` instance.
        """
        return StratigraphyCube(DataCubeInstance,
                                varset=DataCubeInstance.varset,
                                stratigraphy_from=stratigraphy_from,
                                sigma_dist=sigma_dist, dz=dz, z=z, nz=nz)

    def __init__(self, data, read=[], varset=None,
                 stratigraphy_from=None, sigma_dist=None,
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
        super().__init__(data, read, varset)
        if isinstance(data, str):
            raise NotImplementedError('Precomputed NetCDF?')
        elif isinstance(data, np.ndarray):
            raise NotImplementedError('Precomputed numpy array?')
        elif isinstance(data, DataCube):
            # i.e., creating from a DataCube
            _elev = copy.deepcopy(data[stratigraphy_from])

            # set up coordinates of the array
            if sigma_dist is not None:
                _elev_adj = strat._adjust_elevation_by_subsidence(
                    _elev.data, sigma_dist)
            else:
                _elev_adj = _elev.data
            _z = strat._determine_strat_coordinates(
                _elev_adj, dz=dz, z=z, nz=nz)
            self._z = xr.DataArray(_z, name='z', dims=['z'], coords={'z': _z})
            self._H = len(self.z)
            self._L, self._W = _elev.shape[1:]
            self._Z = np.tile(self.z, (self.W, self.L, 1)).T
            self._sigma_dist = sigma_dist

            _out = strat.compute_boxy_stratigraphy_coordinates(
                _elev_adj, sigma_dist=None, z=_z, return_strata=True)
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
            #   use the name of the first dimension, to enable
            #   unlabeled np.ndarrays and flexible name for time
            dim0_name = self.dataio.dims[0]
            dim0_coord = np.array(self.dataio[dim0_name])
            _t = np.expand_dims(dim0_coord,
                                axis=(1, 2))
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

    @property
    def sigma_dist(self):
        """Subsidence information."""
        return self._sigma_dist
