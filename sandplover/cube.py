import abc
import copy
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from sandplover.io import DictionaryIO
from sandplover.io import NetCDFIO
from sandplover.plan import BasePlanform
from sandplover.plan import Planform
from sandplover.plot import VariableSet
from sandplover.section import BaseSection
from sandplover.section import DipSection
from sandplover.section import StrikeSection
from sandplover.strat import BoxyStratigraphyAttributes
from sandplover.strat import MeshStratigraphyAttributes
from sandplover.strat import _adjust_elevation_by_subsidence
from sandplover.strat import _determine_strat_coordinates
from sandplover.strat import compute_boxy_stratigraphy_coordinates
from sandplover.utils import NoStratigraphyError


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

    def __init__(self, data, auxdata=None, read=(), varset=None, dimensions=None):
        """Initialize the BaseCube.

        Parameters
        ----------
        data : :obj:`str`, :obj:`dict`
            If data is type `str`, the string points to a NetCDF or HDF5 file
            that can be read. Typically this is used to directly import files
            output from the pyDeltaRCM model. Alternatively, pass a
            :obj:`dict` with keys indicating variable names, and values with
            corresponding t-x-y `ndarray` of data.

        auxdata : :obj:`str`, optional
            The information `data` is searched for a key matching the string
            `auxdata`, and if found, this key is assigned to `cube.aux`.

        read : :obj:`bool`, optional
            Which variables to read from dataset into memory. Special option
            for ``read=True`` to read all available variables into memory.

        varset : :class:`~sandplover.plot.VariableSet`, optional
            Pass a `~sandplover.plot.VariableSet` instance if you wish
            to style this cube similarly to another cube.

        dimensions : `dict`, optional
            A dictionary with names and coordinates for dimensions of the
            cube, if instantiating the cube from data loaded in memory
            in a dictionary.
        """
        if type(data) is str:
            # handle a path to netCDF file
            self._data_path = data
            self._dataio = NetCDFIO(data_path=data, auxdata_path=auxdata)
            self._read_coords_dims_variables_from_dataio()
        elif type(data) is dict:
            # handle a dict, arrays set up already, make an io class to wrap it
            self._data_path = None
            self._dataio = DictionaryIO(
                data, dimensions=dimensions, auxdata_path=auxdata
            )
            self._read_coords_dims_variables_from_dataio()
        elif isinstance(data, DataCube):
            # handle initializing one cube type from another
            self._data_path = data.data_path
            self._dataio = data._dataio
            self._read_coords_dims_variables_from_dataio()
        else:
            raise TypeError('Invalid type for "data": %s' % type(data))

        self._planform_set = {}  # registered planforms
        self._section_set = {}  # registered sections

        self._registered_variables = []  # list of names registered variables

        if varset:
            self.varset = varset
        else:
            self.varset = VariableSet()

        # some undocumented aliases
        self.plans = self._planform_set
        self.plan_set = self._planform_set

        # Actually use the read parameter to load variables into memory
        if read:
            self.read(read)

    @abc.abstractmethod
    def __getitem__(self, var):
        """Return the variable.

        Overload slicing operations for io to return correct var. Must be
        implemented by subclasses.
        """
        ...

    def _read_coords_dims_variables_from_dataio(self):
        """Read coordinate and dimension information from variables in file.

        Robustly determine dimension names by preferring explicitly
        provided dims, otherwise by scanning for the first 3-D data variable.
        """
        self._coords = self._dataio.known_coords

        # 1) Determine (d0, d1, d2)
        if hasattr(self._dataio, "dims") and len(self._dataio.dims) >= 3:
            d0, d1, d2 = tuple(self._dataio.dims[:3])
        else:
            d0 = d1 = d2 = None
            # Find the first true 3-D variable and use its dims
            for v in self._dataio.known_variables:
                try:
                    da = self.dataio[v]  # xarray.DataArray
                except Exception:
                    continue
                if hasattr(da, "dims") and len(da.dims) == 3:
                    d0, d1, d2 = da.dims
                    break
            if d0 is None:
                raise ValueError(
                    "Could not infer 3-D dimensions from the dataset. "
                    "Provide `dimensions=` when constructing the cube, "
                    "or include at least one 3-D variable."
                )

        # 2) Build coordinate index arrays
        self._dim0_idx = self._dataio[d0]

        # If second coordinate is a 2-D mesh, collapse to 1-D along each axis
        if np.ndim(self._dataio[d1]) == 2:
            self._dim1_idx = self._dataio.dataset[d1][:, 0].squeeze()
            self._dim2_idx = self._dataio.dataset[d2][0, :].squeeze()
        # If second coordinate is 1-D, use both directly
        elif np.ndim(self._dataio[d1]) == 1:
            self._dim1_idx = self._dataio[d1]
            self._dim2_idx = self._dataio[d2]
        else:
            raise TypeError(
                "Shape of coordinate array was not 1-D or 2-D. "
                "Maybe the name was not correctly identified in the dataset, "
                "or the array is misformatted."
            )

        # 3) Expose coords
        self._dim0_coords = self._t = self._dim0_idx
        self._dim1_coords = self._dim1_idx
        self._dim2_coords = self._dim2_idx

        # DEVELOPER NOTE: can we remvoe the _dimX_idx altogether and just use
        # the _dimX_coords arrays?

    def read(self, variables, force=False):
        """Read variable into memory.

        Parameters
        ----------
        variables : :obj:`list` of :obj:`str`, :obj:`str`, :obj:`bool`
            Which variables to read into memory. Pass `True` to read all
            available variables.

        force : `bool`, optional
            If True, bypass memory safety checks and load the data regardless
            of size. Default is False.

        Warnings
        --------
        If any variable size exceeds 80% of currently available RAM and
        `force=False`, a warning will be issued for that variable and it will
        NOT be loaded into memory. Set `force=True` to override this check.
        """
        if variables is True:  # special case, read all variables
            variables = self.dataio.known_variables
        elif type(variables) is str:
            variables = [variables]
        else:
            raise TypeError('Invalid type for "variables": %s ' % variables)

        for var in variables:
            self._dataio.read(var, force=force)

    @property
    def meta(self):
        warnings.warn(
            DeprecationWarning(
                "The `meta` property of the Cube has been replaced by the "
                "`aux` property, and will be removed in a future release."
            )
        )
        return self._dataio.aux

    @property
    def aux(self):
        return self._dataio.aux

    @property
    def auxdata(self):
        """simple alias"""
        return self._dataio.aux

    @property
    def varset(self):
        """:class:`~sandplover.plot.VariableSet` : Variable styling for plotting.

        Can be set with :code:`cube.varset = VariableSetInstance` where
        ``VariableSetInstance`` is a valid instance of
        :class:`~sandplover.plot.VariableSet`.
        """
        return self._varset

    @varset.setter
    def varset(self, var):
        if type(var) is VariableSet:
            self._varset = var
        else:
            raise TypeError("Pass a valid VariableSet instance.")

    @property
    def data_path(self):
        """:obj:`str` : Path connected to for file IO.

        Returns a string if connected to file, or None if cube initialized
        from ``dict``.
        """
        return self._data_path

    @property
    def dataio(self):
        """:obj:`~sandplover.io.BaseIO` subclass : Data I/O handler."""
        return self._dataio

    @property
    def coords(self):
        """`list` : List of coordinate names as strings."""
        return self._coords

    @property
    @abc.abstractmethod
    def variables(self):
        """`list` : List of variable names as strings."""
        ...

    @property
    def registered_variables(self):
        """`list` : List of variable names as strings, subset to those registered (i.e., not in the underlying data)."""
        return self._registered_variables

    @property
    def planform_set(self):
        """:obj:`dict` : Set of planform instances."""
        return self._planform_set

    @property
    def planforms(self):
        """`dict` : Set of plan instances.

        Alias to :meth:`planform_set`.
        """
        return self._planform_set

    def set_aux(self, auxdata):
        """Set a group of the DataIO layer as the 'aux' group."""
        self.dataio._set_aux(auxdata)

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

        PlanformInstance : :obj:`~sandplover.planform.BasePlanform` subclass instance
            The planform instance that will be registered.

        return_planform : :obj:`bool`
            Whether to return the planform object.
        """
        if not issubclass(type(PlanformInstance), BasePlanform):
            raise TypeError(
                "`PlanformInstance` was not a `Planform`. "
                "Instead, was: {}".format(type(PlanformInstance))
            )
        if not isinstance(name, str):
            raise TypeError(
                "`name` was not a string. " "Instead, was: {}".format(type(name))
            )
        PlanformInstance.connect(self, name=name)  # attach cube
        self._planform_set[name] = PlanformInstance
        if return_planform:
            return self._planform_set[name]

    @property
    def section_set(self):
        """:obj:`dict` : Set of section instances."""
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

        SectionInstance : :obj:`~sandplover.section.BaseSection` subclass instance
            The section instance that will be registered.

        return_section : :obj:`bool`
            Whether to return the section object.

        Notes
        -----

        When the API for instantiation of the different section types is
        settled, we should enable the ability to pass section kwargs to this
        method, and then instantiate the section internally. This avoids the
        user having to specify ``spl.section.StrikeSection(distance=2000)`` in
        the ``register_section()`` call, and instead can do something like
        ``golf.register_section('trial', trace='strike',
        distance=2000)``.
        """
        if not issubclass(type(SectionInstance), BaseSection):
            raise TypeError(
                "`SectionInstance` was not a `Section`. "
                "Instead, was: {}".format(type(SectionInstance))
            )
        if not isinstance(name, str):
            raise TypeError(
                "`name` was not a string. " "Instead, was: {}".format(type(name))
            )
        SectionInstance.connect(self, name=name)  # attach cube
        self._section_set[name] = SectionInstance
        if return_section:
            return self._section_set[name]

    def register_variable(self, name, data):
        """Register a variable to the cube.

        Add a variable to the cube, which can be accessed and sliced identically
        to underlying data.

        This is generally useful for data that are derived in the course of
        analyses (i.e., not a part of the original dataset) but are useful (in a
        general sense) for subsequent analyses.

        Only variables with identical dimensionality to the underlying dataset
        can be registered. Registered variables should not be modified after
        registration, but can be easily used in subsequent analysis.

        .. important::

            Registered variables are stored in memory, and are not recorded to
            underlying data!

        Parameters
        ----------
        name : :obj:`str`
            The name to register the variable.

        data : :obj:`xr.DataArray` or :obj:`np.ndarray`
            The data to register. Must have same dimensionality as underying
            data variables.

        Examples
        --------
        See the example document :doc:`/guides/examples/create_from/register_variable`.

        >>> from sandplover.sample_data.sample_data import golf

        >>> golfcube = golf()
        >>> golfcube.register_variable("somevar", np.zeros(golfcube.shape))

        A list of registered variables can be accessed with:

        >>> golfcube.registered_variables
        ['somevar']
        """
        if not isinstance(name, str):
            raise TypeError(f"Input 'name' was not a string, but was {type(name)}")

        # verify shape is identical
        if np.all(data.shape != self.shape):
            raise ValueError(
                f"Input 'data' was incorrect shape {data.shape}. "
                f"Must match cube shape {self.shape}."
            )

        if isinstance(data, np.ndarray):
            # convert to xarray
            data = xr.DataArray(
                data, coords=self._view_coordinates, dims=self._view_dimensions
            )

        # pass to dataio layer to add as needed
        self.dataio._register_variable(name, data)
        # append to list of registered variables
        self._registered_variables.append(name)

    @property
    def dim0_coords(self):
        """Coordinates along the first dimension of `cube`."""
        return self.z

    @property
    def dim1_coords(self):
        """Coordinates along the second dimension of `cube`."""
        return self._dim1_coords

    @property
    def dim2_coords(self):
        """Coordinates along the third dimension of `cube`."""
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
            self.dim2_coords[0],  # dim1, 0
            self.dim2_coords[-1] + self.dim2_coords[1],  # dim1, end + dx
            self.dim1_coords[-1] + self.dim1_coords[1],  # dim0, end + dx
            self.dim1_coords[0],
        ]  # dim0, 0
        _extent = [float(e) for e in _extent]  # quickfix list comp
        return _extent

    @property
    def extent_flipud(self):
        """The `extent`, reversed up-down for special plotting.

        limits of the dim1 by dim2 plane,
        """
        _extent = self.extent
        return [*_extent[:2], _extent[3], _extent[2]]

    @property
    def extent_zeros(self):
        """A dummy variable of zeros with shape of `extent`.

        Returns an array of zeros with the shape of the cube.
        """
        return np.zeros((len(self._dim1_coords), len(self._dim2_coords)), dtype=float)

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
            to :meth:`~sandplover.plan.Planform.show` if `axis` is ``0``,
            otherwise passed
            to :meth:`~sandplover.section.BaseSection.show`.

        Examples
        --------

        .. plot::

            >>> import matplotlib.pyplot as plt
            >>> from sandplover.cube import StratigraphyCube
            >>> from sandplover.sample_data.sample_data import golf

            >>> golfcube = golf()
            >>> golfstrat = StratigraphyCube.from_DataCube(golfcube, dz=0.1)
            >>> fig, ax = plt.subplots(2, 1)
            >>> golfcube.quick_show("eta", ax=ax[0])  # a Planform (axis=0)
            >>> golfstrat.quick_show("eta", idx=100, axis=2, ax=ax[1])  # a DipSection
        """
        if axis == 0:
            # this is a planform slice
            _obj = Planform(self, idx=idx)
        elif axis == 1:
            # this is a Strike section
            _obj = StrikeSection(self, distance_idx=idx)
        elif axis == 2:
            # this is a Dip section
            _obj = DipSection(self, distance_idx=idx)
        else:
            raise ValueError(f"Invalid `axis` specified: {axis}")

        # use the object to handle the showing
        _obj.show(var, **kwargs)

    def show_cube(self, var, style="mesh", ve=200, ax=None):
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

        >>> import matplotlib.pyplot as plt
        >>> from sandplover.cube import StratigraphyCube
        >>> from sandplover.sample_data.sample_data import golf

        >>> golfcube = golf()
        >>> golfstrat = StratigraphyCube.from_DataCube(golfcube, dz=0.1)
        >>> fig, ax = plt.subplots()
        >>> golfstrat.show_cube("eta", ax=ax)  # doctest: +SKIP

        >>> golfcube = golf()
        >>> fig, ax = plt.subplots()
        >>> golfcube.show_cube("velocity", style="fence", ax=ax)  # doctest: +SKIP
        """
        try:
            import pyvista as pv
        except ImportError:
            ImportError("3d plotting dependency, pyvista, was not found.")
        except ModuleNotFoundError:
            ModuleNotFoundError("3d plotting dependency, pyvista, was not found.")
        except Exception as e:
            raise e

        if not ax:
            ax = plt.gca()

        _data = np.array(self[var])
        _data = _data.transpose((2, 1, 0))

        mesh = pv.UniformGrid(_data.shape)
        mesh[var] = _data.ravel(order="F")
        mesh.spacing = (
            self.dim2_coords[1],
            self.dim1_coords[1],
            ve / self.dim1_coords[1],
        )
        mesh.active_scalars_name = var

        p = pv.Plotter()
        p.add_mesh(mesh.outline(), color="k")
        if style == "mesh":
            threshed = mesh.threshold([-np.inf, np.inf], all_scalars=True)
            p.add_mesh(threshed, cmap=self.varset[var].cmap)

        elif style == "fence":
            # todo, improve this to manually create the sections so you can
            #   do more than three slices
            slices = mesh.slice_orthogonal()
            p.add_mesh(slices, cmap=self.varset[var].cmap)

        else:
            raise ValueError(f"Bad value for style: {style}")

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
            "`show_plan` is a deprecated method, and has been replaced by two "
            "alternatives. To quickly show a planform slice of a cube, you "
            "can use `quick_show()` with a similar API. The `show_planform` "
            "method implements more features, but requires instantiating a "
            "`Planform` object first. Passing arguments to `quick_show`.",
            stacklevel=2,
        )
        # pass `t` arg to `idx` for legacy
        if "t" in kwargs:
            idx = kwargs.pop("t")
            kwargs["idx"] = idx

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
            to :meth:`~sandplover.plan.Planform.show`.
        """
        # call `show()` from string
        if isinstance(name, str):
            self._planform_set[name].show(variable, **kwargs)
        else:
            raise TypeError("`name` was not a string, " "was {}".format(type(name)))

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
            to :meth:`~sandplover.section.BaseSection.show`.
        """
        # call `show()` from string
        if isinstance(name, str):
            self._section_set[name].show(variable, **kwargs)
        else:
            raise TypeError("`name` was not a string, " "was {}".format(type(name)))


class DataCube(BaseCube):
    """DataCube object.

    DataCube contains t-x-y information. It may have any
    number of attached attributes (grain size, mud frac, elevation).
    """

    def __init__(
        self,
        data,
        auxdata=None,
        read=(),
        varset=None,
        stratigraphy_from=None,
        dimensions=None,
    ):
        """Initialize the BaseCube.

        Parameters
        ----------
        data : :obj:`str`, :obj:`dict`
            If data is type `str`, the string points to a NetCDF or HDF5 file
            that can be read. Typically this is used to directly import files
            output from the pyDeltaRCM model. Alternatively, pass a :obj:`dict`
            with keys indicating variable names, and values with corresponding
            t-x-y `ndarray` of data.

        auxdata : :obj:`str`, :obj:`dict`, optional
            If `data` is a `str` pointing to a file, then `auxdata` shall be a
            string specifying a group within the file with auxiliary
            information. If `data` is a dictionary, `auxdata` may be either
            another dictionary with auxiliary information, or a `str` specifying
            a key within `data` to be treated as auxiliary informationl; note
            that the last case does not remove the key from `data` variables.
            Default is `None`, no auxiliary information.

        read : :obj:`bool`, optional
            Which variables to read from dataset into memory. Special option for
            ``read=True`` to read all available variables into memory.

        varset : :class:`~sandplover.plot.VariableSet`, optional
            Pass a `~sandplover.plot.VariableSet` instance if you wish to style
            this cube similarly to another cube. If no argument is supplied, a
            new default VariableSet instance is created.

        stratigraphy_from : :obj:`str`, optional
            Pass a string that matches a variable name in the dataset to compute
            preservation and stratigraphy using that variable as elevation data.
            Typically, this is ``'eta'`` in pyDeltaRCM model outputs.
            Stratigraphy can be computed on an existing data cube with the
            :meth:`~sandplover.cube.DataCube.stratigraphy_from` method.

        dimensions : `dict`, optional
            A dictionary with names and coordinates for dimensions of the
            `DataCube`, if instantiating the cube from data loaded in memory in
            a dictionary.
        """
        super().__init__(data, auxdata, read, varset, dimensions=dimensions)

        # Set up the time mesh (DataCube is t–x–y)
        _, self._T, _ = np.meshgrid(
            self.dim1_coords, self.dim0_coords, self.dim2_coords
        )

        # Establish view dimensions/coordinates used by __getitem__ and plotting
        self._view_dimensions = self._dataio.dims
        self._view_coordinates = copy.deepcopy(
            {
                self._view_dimensions[0]: self.dim0_coords,
                self._view_dimensions[1]: self.dim1_coords,
                self._view_dimensions[2]: self.dim2_coords,
            }
        )

        # IMPORTANT: derive shape strictly from coordinates (not variable order/names)
        self._H = int(len(self.dim0_coords))
        self._L = int(len(self.dim1_coords))
        self._W = int(len(self.dim2_coords))

        # Optional stratigraphy bootstrap
        self._knows_stratigraphy = False
        if stratigraphy_from:
            self.stratigraphy_from(variable=stratigraphy_from)

    def __getitem__(self, var):
        """Return the variable.

        Overload slicing operations for io to return a
        :obj:`~sandplover.cube.CubeVariable` instance when slicing.

        Parameters
        ----------
        var : :obj:`str`
            Which variable to slice.

        Returns
        -------
        CubeVariable : `~sandplover.cube.CubeVariable`
            The instantiated CubeVariable.
        """
        # special case for time
        if var == "time":
            # use the name of the first dimension, to enable
            #   unlabeled np.ndarrays and flexible name for time
            dim0_name = self.dataio.dims[0]
            dim0_coord = np.array(self.dataio.dataset[dim0_name])
            _t = np.expand_dims(dim0_coord, axis=(1, 2))
            _obj = xr.DataArray(
                np.tile(_t, (1, *self.shape[1:])),
                coords=self._view_coordinates,
                dims=self._view_dimensions,
            )
        # if the variable is part of the underlying dataio layer
        elif (var in self._coords) or (var in self.dataio._underlying_variables):
            # ensure coords can be called by cube[var]
            _obj = self._dataio[var]
        elif var in self.registered_variables:
            _obj = self._dataio[var]
        else:
            raise AttributeError(f"No variable of '{str(self)}' named '{var}'")

        # make _obj xarray if it not already
        if isinstance(_obj, np.ndarray):
            _obj = xr.DataArray(
                _obj, coords=self._view_coordinates, dims=self._view_dimensions
            )
        return _obj

    def stratigraphy_from(self, variable="eta", style="mesh", **kwargs):
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
            <sandplover.strat.MeshStratigraphyAttributes>` or :obj:`'boxy'
            <sandplover.strat.BoxyStratigraphyAttributes>`. Additional
            keyword arguments are passed to stratigraphy attribute
            initializers.

        **kwargs
            Keyword arguments passed to stratigraphy initialization. Can
            include specification for vertical resolution in `Boxy` case,
            see :obj:_determine_strat_coordinates`.
        """
        if style == "mesh":
            self.strat_attr = MeshStratigraphyAttributes(elev=self[variable], **kwargs)
        elif style == "boxy":
            self.strat_attr = BoxyStratigraphyAttributes(elev=self[variable], **kwargs)
        else:
            raise ValueError('Bad "style" argument supplied: %s' % str(style))
        self._knows_stratigraphy = True

    @property
    def variables(self):
        """Variable available to DataCube.

        Includes only underlying data available from DataIO layer and registered
        variables.
        """
        return self.dataio._underlying_variables + self._registered_variables

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
        """time coordinate.

        This is a one-dimensional array of the time coordinates of the
        `DataCube`.
        """
        return self._t

    @property
    def T(self):
        """Time mesh.

        This is a three-dimensional representation of the time coordinate of
        the `DataCube`. Every element of each row (i.e., layer) of the
        returned array is filled with the corresponding time coordinate
        value.
        """
        return self._T

    @property
    def strata(self):
        if self._knows_stratigraphy:
            return self.strat_attr.strata
        else:
            raise NoStratigraphyError(obj=self, var="strata")


class StratigraphyCube(BaseCube):
    """StratigraphyCube object.

    A cube of precomputed stratigraphy. This is a z-x-y matrix defining
    variables at specific voxel locations.

    This is a special case of a cube.

    """

    @staticmethod
    def from_DataCube(
        DataCubeInstance,
        stratigraphy_from="eta",
        sigma_dist=None,
        dz=None,
        z=None,
        nz=None,
    ):
        """Create from a DataCube.

        Examples
        --------
        Create a stratigraphy cube from the example ``golf``:

        >>> from sandplover.cube import StratigraphyCube
        >>> from sandplover.sample_data.sample_data import golf

        >>> golfcube = golf()
        >>> stratcube = StratigraphyCube.from_DataCube(golfcube, dz=0.05)

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

        sigma_dist : :obj:`float`, :obj:`list` of `float`, optional
            Subsidence distance per timestep or list of subsidence per
            timestep. When a singular (integer or float) value for subsidence
            is provided, it is assumed that the provided value is the rate of
            subsidence in terms of some vertical distance per timestep.
            Conversely, when a time-series is provided, the each value is
            assumed to be the cumulative distance subsided up until that
            point in time. Does not currently support spatially variable subsidence.

        **kwargs
            Keyword arguments passed to stratigraphy initialization. Can
            include specification for vertical resolution in `Boxy` case,
            see :obj:`~sandplover.strat._determine_strat_coordinates`,
            as well as information about subsidence,
            see :obj:`~sandplover.strat._adjust_elevation_by_subsidence`.

        Returns
        -------
        StratigraphyCubeInstance : :obj:`StratigraphyCube`
            The new `StratigraphyCube` instance.
        """
        return StratigraphyCube(
            DataCubeInstance,
            varset=DataCubeInstance.varset,
            stratigraphy_from=stratigraphy_from,
            sigma_dist=sigma_dist,
            dz=dz,
            z=z,
            nz=nz,
        )

    def __init__(
        self,
        data,
        auxdata=None,
        read=(),
        varset=None,
        stratigraphy_from=None,
        sigma_dist=None,
        dz=None,
        z=None,
        nz=None,
    ):
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

        auxdata : :obj:`str`, :obj:`dict`, optional
            If `data` is a `str` pointing to a file, then `auxdata` shall be a
            string specifying a group within the file with auxiliary
            information. If `data` is a dictionary, `auxdata` may be either
            another dictionary with auxiliary information, or a `str` specifying
            a key within `data` to be treated as auxiliary informationl; note
            that the last case does not remove the key from `data` variables.
            Default is `None`, no auxiliary information.

        read : :obj:`bool`, optional
            Which variables to read from dataset into memory. Special option
            for ``read=True`` to read all available variables into memory.

        varset : :class:`~sandplover.plot.VariableSet`, optional
            Pass a `~sandplover.plot.VariableSet` instance if you wish
            to style this cube similarly to another cube. If no argument is
            supplied, a new default VariableSet instance is created.
        """
        super().__init__(data, auxdata, read, varset)
        if isinstance(data, str):
            raise NotImplementedError("Precomputed NetCDF?")
        elif isinstance(data, np.ndarray):
            raise NotImplementedError("Precomputed numpy array?")
        elif isinstance(data, DataCube):
            # i.e., creating from a DataCube
            _elev = copy.deepcopy(data[stratigraphy_from])

            # set up coordinates of the array
            if sigma_dist is not None:
                _elev_adj = _adjust_elevation_by_subsidence(_elev.data, sigma_dist)
            else:
                _elev_adj = _elev.data
            _z = _determine_strat_coordinates(_elev_adj, dz=dz, z=z, nz=nz)
            self._z = xr.DataArray(_z, name="z", dims=["z"], coords={"z": _z})
            self._H = len(self.z)
            self._L, self._W = _elev.shape[1:]
            self._Z = np.tile(self.z, (self.W, self.L, 1)).T
            self._sigma_dist = sigma_dist

            _out = compute_boxy_stratigraphy_coordinates(
                _elev_adj, sigma_dist=None, z=_z, return_strata=True
            )
            self.strata_coords, self.data_coords, self.strata = _out
        else:
            raise TypeError("No other input types implemented yet.")

        self._view_dimensions = ["z", *data._dataio.dims[1:]]
        self._view_coordinates = copy.deepcopy(
            {
                self._view_dimensions[0]: self._z,
                self._view_dimensions[1]: self.dim1_coords,
                self._view_dimensions[2]: self.dim2_coords,
            }
        )

    def __getitem__(self, var):
        """Return the variable.

        Overload slicing operations for io to return a
        :obj:`~sandplover.cube.CubeVariable` instance when slicing, where
        the data have been placed into stratigraphic position.

        Parameters
        ----------
        var : :obj:`str`
            Which variable to slice.

        Returns
        -------
        CubeVariable : `~sandplover.cube.CubeVariable`
            The instantiated CubeVariable.
        """
        if var == "time":
            # a special attribute we add, which matches eta.shape
            #   use the name of the first dimension, to enable
            #   unlabeled np.ndarrays and flexible name for time
            dim0_name = self.dataio.dims[0]
            dim0_coord = np.array(self.dataio[dim0_name])
            _t = np.expand_dims(dim0_coord, axis=(1, 2))
            _arr = np.full(self.shape, np.nan)
            _var = np.tile(_t, (1, *self.shape[1:]))
        elif var in self._dataio.known_variables:
            _arr = np.full(self.shape, np.nan)
            _var = self.dataio[var]
        else:
            raise AttributeError(f"No variable of {str(self)} named {var}")

        # check if the var registered and correct shape, return it
        if var in self.registered_variables:
            if np.all(_var.shape == self.shape):
                return _var
            else:
                raise RuntimeError(
                    f"Registered variable '{var}' sliced, but has incorrect shape: expect {self.shape}, got {_var.shape}"
                )

        # the following lines apply the data to stratigraphy mapping
        if isinstance(_var, xr.core.dataarray.DataArray):
            _vardata = _var.data
        else:
            _vardata = _var
        _cut = _vardata[
            self.data_coords[:, 0], self.data_coords[:, 1], self.data_coords[:, 2]
        ]
        _arr[
            self.strata_coords[:, 0], self.strata_coords[:, 1], self.strata_coords[:, 2]
        ] = _cut
        _obj = xr.DataArray(
            _arr, coords=self._view_coordinates, dims=self._view_dimensions
        )
        return _obj

    @property
    def variables(self):
        """Variable available to StratigraphyCube.

        Includes underlying data available from DataIO layer (including those
        registered to a DataCube sharing the same DataIO layer), and registered
        variables.
        """
        return (
            self.dataio._underlying_variables
            + list(self.dataio._in_memory_variables)
            + self._registered_variables
        )

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
