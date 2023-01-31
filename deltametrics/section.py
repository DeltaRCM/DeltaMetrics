import abc
import warnings

import numpy as np
from scipy import sparse
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from . import cube
from . import plot
from . import mask
from . import plan
from . import utils


@xr.register_dataarray_accessor("strat")
class StratigraphicInformation:
    """Stratigraphic information accessor for SectionVariables.

    Provides an `xarray` accessor called "strat" for holding stratigraphic
    information, and enabling computations and visualizations that depend on
    stratigraphic preservation information.
    """

    _spacetime_names = ["full", "spacetime", "as spacetime", "as_spacetime"]
    _preserved_names = ["psvd", "preserved", "as preserved", "as_preserved"]
    _stratigraphy_names = [
        "strat",
        "strata",
        "stratigraphy",
        "as stratigraphy",
        "as_stratigraphy",
    ]

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._knows_stratigraphy = xarray_obj.knows_stratigraphy
        self._knows_spacetime = xarray_obj.knows_spacetime

    def add_information(self, _psvd_mask=None, _strat_attr=None):
        # check information is valid for object
        if _psvd_mask is not None:
            _psvd_mask = np.asarray(_psvd_mask)
            if _psvd_mask.shape != self._obj.shape:
                raise ValueError(
                    'Shape of "_psvd_mask" incompatible with "_data" array.'
                )
        self._psvd_mask = _psvd_mask

        if not (_strat_attr is None):
            self.strat_attr = _strat_attr
            self._knows_stratigraphy = True
        else:
            self._knows_stratigraphy = False

    def __getitem__(self, slc):
        """Get items from the underlying data.

        Takes a numpy slicing style and slices data from the underlying data.
        Note that the underlying data is stored in an :obj:`xarray.DataArray`,
        and this method returns a :obj:`xarray.DataArray`.

        Parameters
        ----------
        slc : a `numpy` slice
            A valid `numpy` style slice. For example, :code:`[10, ...]`.
            Dimension validation is not performed before slicing.
        """
        return self._obj[slc]

    @property
    def data(self):
        # undocumented helpful wrapper
        return self._obj.data

    @property
    def values(self):
        # undocumented helpful wrapper
        return self._obj.values

    @property
    def knows_spacetime(self):
        """Whether the data variable knows preservation information."""
        return self._knows_spacetime

    def _check_knows_spacetime(self):
        """Check whether "knows_spacetime".

        Raises
        ------
        AttributeError
            Raises if does not know spacetime, otherwise returns
            `self._knows_spacetime` (i.e., `True`).
        """
        if not self._knows_spacetime:
            raise AttributeError('No "spacetime" or "preserved" information available.')
        else:
            return self._knows_spacetime

    @property
    def knows_stratigraphy(self):
        """Whether the data variable knows preservation information."""
        return self._knows_stratigraphy

    def _check_knows_stratigraphy(self):
        """Check whether "knows_stratigraphy".

        Raises
        ------
        AttributeError
            Raises if does not know stratigraphy.
        """
        if not self._knows_stratigraphy:
            raise utils.NoStratigraphyError(obj=self)
        return self._knows_stratigraphy

    def as_preserved(self):
        """Variable with only preserved values.

        Returns
        -------
        ma : :obj:`np.ma.MaskedArray`
            A numpy MaskedArray with non-preserved values masked.
        """
        if self._check_knows_stratigraphy():
            return self._obj.where(self._psvd_mask)

    def as_stratigraphy(self):
        """Variable as preserved stratigraphy.

        .. warning::

            This method returns a sparse array that is not suitable to be
            displayed directly. Use
            :obj:`get_display_arrays(style='stratigraphy')` instead to get
            corresponding x-y coordinates for plotting the array.
        """
        if self._check_knows_stratigraphy():
            # actual data, where preserved
            _psvd_data = self._obj.data[self.strat_attr["psvd_idx"]]
            _sp = sparse.coo_matrix(
                (_psvd_data, (self.strat_attr["z_sp"], self.strat_attr["s_sp"]))
            )
            return _sp


class BaseSection(abc.ABC):
    """Base section object.

    Defines common attributes and methods of a section object.

    This object should wrap around many of the functions available from
    :obj:`~deltametrics.strat`.

    """

    def __init__(self, section_type, *args, name=None):
        """
        Identify coordinates defining the section.

        Parameters
        ----------
        section_type : :obj:`str`
            String identifying the *type* of `Section` being instantiated.

        CubeInstance : :obj:`~deltametrics.cube.BaseCube` subclass, optional
            Connect to this cube. No connection is made if cube is not
            provided.

        name : :obj:`str`, optional
            An optional name for the `Section` object, helpful for maintaining
            and keeping track of multiple `Section` objects of the same
            type. This is disctinct from the :obj:`section_type`. The name
            is used internally if you use the :obj:`register_section` method
            of a `Cube`. Notes

        Notes
        -----

        If no arguments are passed, an empty section not connected to any cube
        is returned. This cube will will need to be manually connected to have
        any functionality (via the :meth:`connect` method).
        """
        # begin unconnected
        self._s = None
        self._z = None
        self._dim1_idx = None
        self._dim2_idx = None
        self._trace = None
        self._shape = None
        self._variables = None
        self._underlying = None
        self._underlying_type = None

        self.section_type = section_type
        self._name = name  # default `name` is None

        # check that zero or one postitional argument was given
        if len(args) > 1:
            raise ValueError(
                "Expected single positional argument to \
                             %s instantiation."
                % type(self)
            )

        # if one positional argument was given, connect to the cube,
        #    otherwise return an unconnected section.
        if len(args) > 0:
            self.connect(args[0])
        else:
            pass

    def connect(self, InputInstance, name=None):
        """Connect this Section instance to a Cube instance."""
        if issubclass(type(InputInstance), cube.BaseCube):
            self._underlying = InputInstance
            self._underlying_type = "cube"
            self._variables = InputInstance.variables
            self._L = InputInstance.L
            self._W = InputInstance.W
            self._underlying_dim1_coords = InputInstance.dim1_coords
            self._underlying_dim2_coords = InputInstance.dim2_coords
            self._z = self._underlying.z
        elif issubclass(type(InputInstance), mask.BaseMask):
            self._underlying = InputInstance
            self._underlying_type = "mask"
            self._variables = None
            self._L = InputInstance.shape[0]
            self._W = InputInstance.shape[1]
            _mask_xarray = InputInstance._mask
            self._underlying_dim1_coords = _mask_xarray[_mask_xarray.dims[0]]
            self._underlying_dim2_coords = _mask_xarray[_mask_xarray.dims[1]]
            self._z = None
        elif utils.is_ndarray_or_xarray(InputInstance):
            self._underlying = InputInstance
            self._underlying_type = "array"
            self._variables = None
            self._L = InputInstance.shape[0]
            self._W = InputInstance.shape[1]
            if isinstance(InputInstance, xr.core.dataarray.DataArray):
                self._underlying_dim1_coords = InputInstance[InputInstance.dims[0]]
                self._underlying_dim2_coords = InputInstance[InputInstance.dims[1]]
            elif isinstance(InputInstance, np.ndarray):
                self._underlying_dim1_coords = np.arange(self._L, dtype=float)
                self._underlying_dim2_coords = np.arange(self._W, dtype=float)
            self._z = None
        elif issubclass(type(InputInstance), plan.Planform):
            self._underlying = InputInstance
            self._underlying_type = "planform"
            self._variables = InputInstance.variables
            self._L = InputInstance.shape[0]
            self._W = InputInstance.shape[1]
            _plan_xarray = InputInstance[self._variables[0]]
            self._underlying_dim1_coords = _plan_xarray[_plan_xarray.dims[0]]
            self._underlying_dim2_coords = _plan_xarray[_plan_xarray.dims[1]]
            self._z = None  # take from plan?
        else:
            raise TypeError(
                "Expected type is subclass of Cube, Planform, or Mask, "
                "but received was {_gottype}.".format(_gottype=type(InputInstance))
            )

        self.name = name  # use the setter to determine the _name
        self._compute_section_coords()
        self._compute_section_attrs()

    @property
    def name(self):
        """Section name.

        Helpful to differentiate multiple `Section` objects.
        """
        return self._name

    @name.setter
    def name(self, var):
        if self._name is None:
            # _name is not yet set
            self._name = var or self.section_type
        else:
            # _name is already set
            if not (var is None):
                warnings.warn(
                    UserWarning(
                        "`name` argument supplied to instantiated "
                        "`Section` object. To change the name of "
                        "a Section, you must set the attribute "
                        "directly with `section._name = 'name'`."
                    )
                )
            # do nothing

    @abc.abstractmethod
    def _compute_section_coords(self):
        """Should calculate x-y coordinates of the section.

        Sets the value ``self._dim1_idx`` and ``self._dim2_idx`` according to
        the algorithm of each section initialization.

        .. warning::

            When implementing new section types, be sure that
            ``self._dim1_idx`` and ``self._dim2_idx`` are *one-dimensional
            arrays*, or you will get an improperly shaped Section array in
            return.
        """
        ...

    def _compute_section_attrs(self):
        """Compute attrs

        Compute the along-section coordinate array from dimensions in the
        cube (`dim1`, `dim2`) definining the section.
        """
        underlying_dx = float(
            self._underlying_dim1_coords[1] - self._underlying_dim1_coords[0]
        )
        self._trace_idx = np.column_stack((self._dim1_idx, self._dim2_idx))
        self._trace = np.column_stack(
            (
                self._underlying_dim2_coords[self._dim2_idx] + underlying_dx / 2,
                self._underlying_dim1_coords[self._dim1_idx] + underlying_dx / 2,
            )
        )

        # compute along section distance and place into a DataArray
        _s = np.cumsum(
            np.hstack(
                (
                    0,
                    np.sqrt(
                        (self._trace[1:, 0] - self._trace[:-1, 0]) ** 2
                        + (self._trace[1:, 1] - self._trace[:-1, 1]) ** 2
                    ),
                )
            )
        )
        self._s = xr.DataArray(_s, name="s", dims=["s"], coords={"s": _s})
        self._length = float(_s[-1]) + underlying_dx

        # set shape from the coordinates
        if not (self._z is None):
            self._shape = (len(self._z), len(self._s))
        else:
            self._shape = (len(self._s),)

    @property
    def idx_trace(self):
        """Alias for `self.trace_idx`."""
        return self._trace_idx

    @property
    def trace_idx(self):
        """Indices of section points in the `dim1`-`dim2` plane."""
        return self._trace_idx

    @property
    def trace(self):
        """Coordinates of the section in the `dim1`-`dim2` plane.

        .. note:: stack of [dim2, dim1].
        """
        return self._trace

    @property
    def s(self):
        """Along-section coordinate."""
        return self._s

    @property
    def length(self):
        """Length of section in dimensional coordinates."""
        return self._length

    @property
    def z(self):
        """Up-section (vertical) coordinate."""
        return self._z

    @property
    def shape(self):
        """Section shape.

        Simply a `tuple` equivalent to ``(len(z), len(s))``
        """
        return self._shape

    @property
    def variables(self):
        """List of variables."""
        return self._variables

    @property
    def strat_attr(self):
        """Stratigraphic attributes data object.

        Raises
        ------
        NoStratigraphyError
            If no stratigraphy information is found for the section.
        """
        if self._underlying._knows_stratigraphy:
            return self._underlying.strat_attr
        else:
            raise utils.NoStratigraphyError(obj=self, var="strat_attr")

    def __getitem__(self, var):
        """Get a slice of the section.

        Slicing the section instance creates an `xarray` `DataArray` instance
        from data, for variable ``var`` and maintaining the data coordinates.

        .. note:: We only support slicing by string.

        Parameters
        ----------
        var : :obj:`str`
            Which variable to slice.

        Returns
        -------
        data : :obj:`DataArray`
            The underlying data returned as an xarray `DataArray`, maintaining
            coordinates.
        """
        if self._underlying_type == "cube":
            if isinstance(self._underlying, cube.DataCube):
                _xrDA = xr.DataArray(
                    self._underlying[var].data[:, self._dim1_idx, self._dim2_idx],
                    coords={"s": self._s, self._z.dims[0]: self._z},
                    dims=[self._z.dims[0], "s"],
                    name=var,
                    attrs={
                        "slicetype": "data_section",
                        "knows_stratigraphy": self._underlying._knows_stratigraphy,
                        "knows_spacetime": True,
                    },
                )
                if self._underlying._knows_stratigraphy:
                    _xrDA.strat.add_information(
                        _psvd_mask=self._underlying.strat_attr.psvd_idx[
                            :, self._dim1_idx, self._dim2_idx
                        ],  # noqa: E501
                        _strat_attr=self._underlying.strat_attr(
                            "section", self._dim1_idx, self._dim2_idx
                        ),
                    )
                return _xrDA
            elif isinstance(self._underlying, cube.StratigraphyCube):
                _xrDA = xr.DataArray(
                    self._underlying[var].data[:, self._dim1_idx, self._dim2_idx],
                    coords={"s": self._s, self._z.dims[0]: self._z},
                    dims=[self._z.dims[0], "s"],
                    name=var,
                    attrs={
                        "slicetype": "stratigraphy_section",
                        "knows_stratigraphy": True,
                        "knows_spacetime": False,
                    },
                )
                return _xrDA
            else:
                raise TypeError(
                    "Unknown Cube type encountered: %s" % type(self._underlying)
                )
        elif self._underlying_type in ["mask", "planform"]:
            _xrDA = xr.DataArray(
                self._underlying[var].data[self._dim1_idx, self._dim2_idx],
                coords={"s": self._s},
                dims=["s"],
                name=var,
            )
            return _xrDA
        elif self._underlying_type == "array":
            _xrDA = xr.DataArray(
                self._underlying[self._dim1_idx, self._dim2_idx],
                coords={"s": self._s},
                dims=["s"],
                name=var,
            )
            return _xrDA
        elif self._underlying is None:
            raise AttributeError(
                "No underlying data connected. Are you sure you ran `.connect()`?"
            )
        else:
            raise TypeError(
                f"Unknown underlying type string encountered "
                f"{str(self._underlying_type)}. Type of underlying is "
                f"{type(self._underlying)}."
            )

    def show(
        self,
        *args,
        style="shaded",
        data=None,
        label=False,
        colorbar=True,
        colorbar_label=False,
        ax=None,
    ):
        """Show the section.

        Method enumerates convenient routines for visualizing sections of data
        and stratigraphy. Includes support for multiple data `style` and
        multiple `data` choices as well.

        .. note::

            The colors for `style='lines'` are determined from the left-end
            edge node, and colors for the `style='shaded'` mesh are determined
            from the lower-left-end edge node of the quad.

        Parameters
        ----------
        SectionAttribute : :obj:`str`, :obj:`SectionVariableInstance`
            Which attribute to show. Can be a string for a named `Cube`
            attribute, or any arbitrary data. Additionally, pass no arguments
            and the first variable in the underlying data source list will be
            used.

        style : :obj:`str`, optional
            What style to display the section with. Choices are 'mesh' or
            'line'.

        data : :obj:`str`, optional
            Argument passed to
            :obj:`~deltametrics.section.DataSectionVariable.get_display_arrays`
            or
            :obj:`~deltametrics.section.DataSectionVariable.get_display_lines`.
            Supported options are `'spacetime'`, `'preserved'`, and
            `'stratigraphy'`. Default is to display full spacetime plot for
            section generated from a `DataCube`, and stratigraphy for a
            `StratigraphyCube` section.

        label : :obj:`bool`, `str`, optional
            Display a label of the variable name on the plot. Default is
            False, display nothing. If ``label=True``, the label name from the
            :obj:`~deltametrics.plot.VariableSet` is used. Other arguments are
            attempted to coerce to `str`, and the literal is diplayed.

        colorbar : :obj:`bool`, optional
            Whether a colorbar is appended to the axis.

        colorbar_label : :obj:`bool`, `str`, optional
            Display a label of the variable name along the colorbar. Default is
            False, display nothing. If ``label=True``, the label name from the
            :obj:`~deltametrics.plot.VariableSet` is used. Other arguments are
            attempted to coerce to `str`, and the literal is diplayed.

        ax : :obj:`~matplotlib.pyplot.Axes` object, optional
            A `matplotlib` `Axes` object to plot the section. Optional; if not
            provided, a call is made to ``plt.gca()`` to get the current (or
            create a new) `Axes` object.

        Examples
        --------
        *Example 1:* Display the `velocity` spacetime section of a DataCube.

        .. doctest::

            >>> golfcube = dm.sample_data.golf()
            >>> golfcube.register_section(
            ...     'demo', dm.section.StrikeSection(distance_idx=5))
            >>> golfcube.sections['demo'].show('velocity')

        .. plot:: section/section_demo_spacetime.py

        Note that the last line above is functionally equivalent to
        ``golfcube.show_section('demo', 'velocity')``.

        *Example 2:* Display a section, with "quick" stratigraphy, as the
        `depth` attribute, displaying several different section styles.

        .. doctest::

            >>> golfcube = dm.sample_data.golf()
            >>> golfcube.stratigraphy_from('eta')
            >>> golfcube.register_section(
            ...     'demo', dm.section.StrikeSection(distance=250))

            >>> fig, ax = plt.subplots(4, 1, sharex=True, figsize=(6, 9))
            >>> golfcube.sections['demo'].show('depth', data='spacetime',
            ...                                 ax=ax[0], label='spacetime')
            >>> golfcube.sections['demo'].show('depth', data='preserved',
            ...                                ax=ax[1], label='preserved')
            >>> golfcube.sections['demo'].show('depth', data='stratigraphy',
            ...                                ax=ax[2], label='quick stratigraphy')
            >>> golfcube.sections['demo'].show('depth', style='lines', data='stratigraphy',
            ...                                ax=ax[3], label='quick stratigraphy')          # noqa: E501

        .. plot:: section/section_demo_quick_strat.py
        """
        # check that someting is attached
        if self._underlying is None:
            raise AttributeError(
                "No underlying data connected. Are you sure you ran `.connect()`?"
            )

        # process arguments and inputs
        if len(args) == 0:
            SectionAttribute = self._underlying.variables[0]
        elif len(args) == 1:
            SectionAttribute = args[0]
        else:
            raise ValueError(
                "Zero or one positional argument must be passed to `show`."
            )
        if not ax:
            ax = plt.gca()

        # work through display options based on type of underlying
        SectionVariableInstance = self[SectionAttribute]
        if self._underlying_type == "cube":
            # if te underlying is a cube
            _varinfo = (
                self._underlying.varset[SectionAttribute]
                if issubclass(type(self._underlying), cube.BaseCube)
                else plot.VariableSet()[SectionAttribute]
            )
            # main routines for plot styles
            if style in ["shade", "shaded"]:
                _data, _X, _Y = plot.get_display_arrays(
                    SectionVariableInstance, data=data
                )
                ci = ax.pcolormesh(
                    _X,
                    _Y,
                    _data,
                    cmap=_varinfo.cmap,
                    norm=_varinfo.norm,
                    vmin=_varinfo.vmin,
                    vmax=_varinfo.vmax,
                    shading="flat",
                    rasterized=True,
                )
            elif style in ["line", "lines"]:
                _data, _segments = plot.get_display_lines(
                    SectionVariableInstance, data=data
                )
                lc = LineCollection(_segments, cmap=_varinfo.cmap)
                lc.set_array(_data.flatten())
                lc.set_linewidth(1.25)
                ci = ax.add_collection(lc)
            else:
                raise ValueError('Bad style argument: "%s"' % style)

            # style adjustments
            if colorbar:
                cb = plot.append_colorbar(ci, ax)
                if colorbar_label:
                    _colorbar_label = (
                        _varinfo.label
                        if (colorbar_label is True)
                        else str(colorbar_label)
                    )  # use custom if passed
                    cb.ax.set_ylabel(_colorbar_label, rotation=-90, va="bottom")
            ax.margins(y=0.2)
            if label:
                _label = (
                    _varinfo.label if (label is True) else str(label)
                )  # use custom if passed
                ax.text(
                    0.99,
                    0.8,
                    _label,
                    fontsize=10,
                    horizontalalignment="right",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )

            # set the limits of the plot accordingly
            xmin, xmax, ymin, ymax = plot.get_display_limits(
                SectionVariableInstance, data=data
            )
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        else:
            SectionVariableInstance = self[SectionAttribute]
            ax.plot(SectionVariableInstance)
            # raise NotImplementedError()

    def show_trace(self, *args, ax=None, autoscale=False, **kwargs):
        """Plot section trace (x-y plane path).

        Plot the section trace (:obj:`trace`) onto an x-y planview.

        Parameters
        ----------
        *args
            Passed to `matplotlib` :obj:`~matplotlib.pyplot.plot()`.

        ax : :obj:`~matplotlib.pyplot.Axes` object, optional
            A `matplotlib` `Axes` object to plot the trace. Optional; if not
            provided, a call is made to ``plt.gca()`` to get the current (or
            create a new) `Axes` object.

        autoscale : :obj:`bool`
            Whether to rescale the axis based on the limits of the section.
            Manipulates the `matplotlib` `autoscale` attribute. Default is
            ``False``.

        **kwargs
            Passed to `matplotlib` :obj:`~matplotlib.pyplot.plot()`.
        """
        if not ax:
            ax = plt.gca()

        _label = kwargs.pop("label", self.name)

        # _x = self._underlying_dim2_coords[self._dim2_idx]
        # _y = self._underlying_dim1_coords[self._dim1_idx]
        _x = self._trace[:, 0]
        _y = self._trace[:, 1]

        # get the limits to be able to reset if autoscale false
        lims = [ax.get_xlim(), ax.get_ylim()]

        # add the trace
        ax.plot(_x, _y, label=_label, *args, **kwargs)

        # if autscale is false, reset the axes
        if not autoscale:
            ax.set_xlim(*lims[0])
            ax.set_ylim(*lims[1])


class PathSection(BaseSection):
    """Path section object.

    Create a Section along user-specified path. Specify the section location
    as an `(N, 2)` `ndarray` of `dim1-dim2` pairs of coordinates that define
    the verticies of the path. All coordinates along the path will be
    included in the section.

    .. plot::

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section('path', dm.section.PathSection(
        ...     path_idx=np.array([[3, 50], [17, 65], [10, 130]])))
        >>> fig, ax = plt.subplots()
        >>> golfcube.quick_show('eta', idx=-1, ax=ax, ticks=True)
        >>> golfcube.sections['path'].show_trace('r--', ax=ax)
        >>> plt.show()

    Parameters
    ----------
    *args : :obj:`DataCube` or `StratigraphyCube`
        The `Cube` object to link for underlying data. This option should be
        ommitted if using the :obj:`register_section` method of a `Cube`.

    path : :obj:`ndarray`
        An `(N, 2)` `ndarray` specifying the `dim1-dim2` pairs of coordinates
        in dimensional values, defining the verticies of the path to extract
        the section from. Mutually exclusive with `path_idx`.

    path_idx : :obj:`ndarray`
        An `(N, 2)` `ndarray` specifying the `dim1-dim2` pairs of coordinates
        in dimension indices, defining the verticies of the path to extract
        the section from. Mutually exclusive with `path`.

    **kwargs
        Keyword arguments are passed to `BaseSection.__init__()`. Supported
        options are `name`.

    Returns
    -------
    section : :obj:`PathSection`
        `PathSection` object with specified parameters. The section is
        automatically connected to the underlying `Cube` data source if the
        :obj:`register_section` method of a `Cube` is used to set up the
        section, or the `Cube` is passed as the first positional argument
        during instantiation.

    Examples
    --------

    Create a `PathSection` that is registered to a `DataCube` at
    specified coordinates:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section('path', dm.section.PathSection(
        ...     path=np.array([[2000, 2000], [2000, 6000], [600, 7500]])))
        >>>
        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.quick_show('eta', idx=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['path'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['path'].show('velocity', ax=ax[1])
        >>> plt.show()

    Create a `PathSection` that is registered to a `DataCube` at
    specified indices:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section('path', dm.section.PathSection(
        ...     path_idx=np.array([[3, 50], [17, 65], [10, 130]])))
        >>>
        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.quick_show('eta', idx=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['path'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['path'].show('velocity', ax=ax[1])
        >>> plt.show()
    """

    def __init__(self, *args, path=None, path_idx=None, **kwargs):
        """Instantiate.

        Parameters
        ----------
        path : :obj:`ndarray`
            An `(N, 2)` `ndarray` specifying the dim1-dim2 pairs of
            coordinates that define the verticies of the path to extract the
            section from.

        .. note::

            :obj:`path` must be supplied as a keyword argument.

        """
        if (path is None) and (path_idx is None):
            raise ValueError("Must specify `path` or `path_idx`.")
        #   if both path and idx are given
        if (not (path is None)) and (not (path_idx is None)):
            raise ValueError("Cannot specify both `path` and `path_idx`.")

        self._input_path = path
        self._input_path_idx = path_idx

        super().__init__("path", *args, **kwargs)

    def _compute_section_coords(self):
        """Calculate coordinates of the strike section."""

        dim1_coords = self._underlying_dim1_coords
        dim2_coords = self._underlying_dim2_coords

        # note: _path is given as N x dim1,dim2
        # if input path is given, we need to convert to indices
        if not (self._input_path is None):
            _dim1_pts = np.argmin(
                np.abs(
                    self._input_path[:, 0]
                    - np.tile(dim1_coords, (self._input_path.shape[0], 1)).T
                ),
                axis=0,
            )
            _dim2_pts = np.argmin(
                np.abs(
                    self._input_path[:, 1]
                    - np.tile(dim2_coords, (self._input_path.shape[0], 1)).T
                ),
                axis=0,
            )
            _path = np.column_stack((_dim1_pts, _dim2_pts))

        # otherwise, the path must be given as indices
        else:
            _path = self._input_path_idx

        # convert the points into segments into lists of cells
        #    input to utils needs to be xy cartesian order
        _segs = utils.coordinates_to_segments(np.fliplr(_path))
        _cell = utils.segments_to_cells(_segs)

        # determine only unique coordinates along the path
        def unsorted_unique(array):
            """internal utility unsorted version of np.unique
            https://stackoverflow.com/a/12927009/
            """
            uniq, index = np.unique(array, return_index=True, axis=0)
            return uniq[index.argsort()]

        # store values
        self._path_idx = unsorted_unique(_cell)
        self._vertices_idx = unsorted_unique(_path)
        self._vertices = np.column_stack(
            (dim1_coords[_path[:, 0]], dim2_coords[_path[:, 1]])
        )

        self._dim1_idx = self._path_idx[:, 1]
        self._dim2_idx = self._path_idx[:, 0]

    @property
    def path(self):
        """Path of the PathSection.

        Returns same as `trace` property.
        """
        return self.trace

    @property
    def vertices(self):
        """Vertices defining the path in dimensional coordinates."""
        return self._vertices


class LineSection(BaseSection):
    def __init__(
        self,
        direction,
        *args,
        distance=None,
        distance_idx=None,
        length=None,
        x=None,
        y=None,
        **kwargs,
    ):
        """Initialization for the LineSection.

        The LineSection is the base class for Strike and Dip sections,
        as these share identical input arguments, and processing steps, but
        differ in which direction the line is drawn.

        .. note:: the `RadialSection` does not subclass `LineSection`.
        """

        self._distance = None
        self._distance_idx = None
        self._length = None
        self._length_idx = None

        # process the optional/deprecated input arguments
        #   if y or x is given, cannot also give distance idx or length
        if (not (y is None)) or (not (x is None)):
            #   check if new args are given
            if (
                (not (distance is None))
                or (not (distance_idx is None))
                or (not (length is None))
            ):  # noqa: E501
                raise ValueError(
                    "Cannot specify `distance`, `distance_idx`, or `length` "
                    "if specifying `y` or `x`."
                )
            #   if new args not given, then use old args in place of new
            else:
                warnings.warn(
                    "Arguments `y` and `x` are deprecated and will be removed"
                    "in a future release. Please use `distance_idx` and "
                    "`length` to continue to specify cell indices, or "
                    "use `distance` and `length` to specify "
                    "coordinate values."
                )
                if direction == "strike":
                    distance_idx = y
                    length = x
                elif direction == "dip":
                    distance_idx = x
                    length = y
                else:
                    raise ValueError("Invalid `direction`.")
        else:
            #   if y or x is not given, must give either distance or idx
            if (distance is None) and (distance_idx is None):
                raise ValueError("Must specify `distance` or `distance_idx`.")
        #   if both distance and idx are given
        if (not (distance is None)) and (not (distance_idx is None)):
            raise ValueError("Cannot specify both `distance` and `distance_idx`.")

        self._input_distance = distance
        self._input_distance_idx = distance_idx
        self._input_length = length
        super().__init__(direction, *args, **kwargs)

    @property
    def distance(self):
        """Distance of section from reference edge, in perpendicular-reference coordinates."""
        return self._distance

    @property
    def distance_idx(self):
        """Distance of section from from reference edge, in perpendicular-reference indices."""
        return self._distance_idx

    # @property
    # def length(self):
    #     """Bounding `dim2` coordinates of section."""
    #     return self._length

    # @property
    # def length_idx(self):
    #     """Bounding `dim2` indices of section."""
    #     return self._length_idx


class StrikeSection(LineSection):
    """Strike section object.

    Section oriented parallel to the `dim2` axis. Specify the location of the
    strike section with :obj:`distance` and :obj:`length` *or*
    :obj:`distance_idx` and :obj:`length` keyword parameters.

    .. plot::

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section(
        ...     'strike', dm.section.StrikeSection(distance=1500))
        >>> fig, ax = plt.subplots()
        >>> golfcube.quick_show('eta', idx=-1, ax=ax, ticks=True)
        >>> golfcube.sections['strike'].show_trace('r--', ax=ax)
        >>> plt.show()

    .. hint::

        Either :obj:`distance` *or* :obj:`distance_idx` must be specified.

    Parameters
    ----------
    *args : :obj:`DataCube` or `StratigraphyCube`
        The `Cube` object to link for underlying data. This option should be
        ommitted if using the :obj:`register_section` method of a `Cube`.

    distance : :obj:`float`, optional
        Distance *in `dim1` coordinates* from the `dim1` lower domain edge to
        place the section. The section location will be interpolated to the
        nearest grid cell. Mutually exclusive with `distance_idx`.

    distance_idx : :obj:`int`, optional
        Distance *in cell indices* from the `dim1` lower domain edge to place
        the section. Mutually exclusive with `distance`.

    length : :obj:`tuple` or :obj:`list` of `int` or `float`, optional
        A two-element tuple specifying the bounding points of the section in
        the `dim2` axis. Values are treated as cell indices
        if :obj:`distance_idx` is given and as `dim2` coordinates
        if :obj:`distance` is given. If no value is supplied, the section is
        drawn across the entire `dim2` axis (i.e., across the whole domain).

    y : :obj:`int`, optional, deprecated
        The number of cells in from the `dim1` lower domain edge. If used, the
        value is internally coerced to :obj:`distance_idx`.

    x : :obj:`int`, optional, deprecated
        The limits of the section. Defaults to the full `dim2`
        domain `width`. Specify as a two-element `tuple` or `list` of `int`,
        giving the lower and upper bounds of indices to span the section.

    **kwargs
        Keyword arguments are passed to `BaseSection.__init__()`. Supported
        options are `name`.

    Returns
    -------
    section : :obj:`StrikeSection`
        `StrikeSection` object with specified parameters. The section is
        automatically connected to the underlying `Cube` data source if the
        :obj:`register_section` method of a `Cube` is used to set up the
        section, or the `Cube` is passed as the first positional argument
        during instantiation.

    Examples
    --------

    Create a `StrikeSection` that is registered to a `DataCube` at specified
    `distance` in `dim1` coordinates, and spans the entire model domain:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section('strike', dm.section.StrikeSection(distance=3500))

        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.quick_show('eta', idx=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['strike'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['strike'].show('velocity', ax=ax[1])
        >>> plt.show()

    Create a `StrikeSection` that is registered to a `DataCube` at
    specified `distance_idx` index ``=10``, and spans the entire model domain:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section('strike', dm.section.StrikeSection(distance_idx=10))

        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.quick_show('eta', idx=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['strike'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['strike'].show('velocity', ax=ax[1])
        >>> plt.show()

    Create a `StrikeSection` that is registered to a `StratigraphyCube` at
    specified `distance` in `dim1` coordinates, and spans only a range in the
    middle of the domain:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfstrat = dm.cube.StratigraphyCube.from_DataCube(golfcube, dz=0.1)
        >>> golfstrat.register_section(
        ...     'strike_part', dm.section.StrikeSection(distance=1500, length=(2000, 5000)))

        >>> # show the location and the "time" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.quick_show('eta', idx=-1, ax=ax[0], ticks=True)
        >>> golfstrat.sections['strike_part'].show_trace('r--', ax=ax[0])
        >>> golfstrat.sections['strike_part'].show('time', ax=ax[1])
        >>> plt.show()
    """

    def __init__(
        self,
        *args,
        distance=None,
        distance_idx=None,
        length=None,
        y=None,
        x=None,
        **kwargs,
    ):
        # initialization is handled by the `LineSection` class
        # _compute_section_coords is called by the `BaseSection` class
        super().__init__(
            "strike",
            *args,
            distance=distance,
            distance_idx=distance_idx,
            length=length,
            x=x,
            y=y,
            **kwargs,
        )

    def _compute_section_coords(self):
        """Calculate coordinates of the strike section."""

        dim1_coords = self._underlying_dim1_coords
        dim2_coords = self._underlying_dim2_coords

        # if input length is None, we need to use endpoints of the dim2 coords
        if self._input_length is None:
            # if the value is given as distance
            if not (self._input_distance is None):
                _length = (
                    float(dim2_coords[0]),
                    float(dim2_coords[-1]),
                )
            # if the value is given as idx
            else:
                _length = (0, self._W - 1)

        else:
            # quick check that value for length is valid
            if len(self._input_length) != 2:
                raise ValueError(
                    "Input `length` must be two element tuple or list, "
                    "but was {0}".format(str(self._input_length))
                )
            _length = self._input_length

        # if the value is given as distance
        if not (self._input_distance is None):
            # interpolate to an idx
            _idx = np.argmin(np.abs(np.array(dim1_coords) - self._input_distance))
            # treat length as coordinates
            #   should have somechecks here for valid values?
            _start_idx = np.argmin(np.abs(np.array(dim2_coords) - _length[0]))
            _end_idx = np.argmin(np.abs(np.array(dim2_coords) - _length[1])) - 1
        else:
            # apply the input idx value
            _idx = int(self._input_distance_idx)
            # treat length as indices
            _start_idx, _end_idx = _length

        self._distance = float(dim1_coords[_idx])
        self._distance_idx = _idx
        self._start_end = _length
        self._start_end_idx = (_start_idx, _end_idx)

        # now compute the indices to use for the section
        self._dim2_idx = np.arange(_start_idx, _end_idx + 1, dtype=int)
        self._dim1_idx = np.tile(self._distance_idx, (len(self._dim2_idx)))
        self._length_idx = len(self._dim2_idx)

    @property
    def y(self):
        """Deprecated. Use :obj:`distance_idx`."""
        warnings.warn("`.y` is a deprecated attribute. Use `.distance_idx` instead.")
        return self._distance_idx

    @property
    def x(self):
        """Deprecated. Use :obj:`length_idx`.

        Start and end indices of section.
        """
        warnings.warn("`.x` is a deprecated attribute. Use `.length_idx` instead.")
        return self._length_idx


class DipSection(LineSection):
    """Dip section object.

    Section oriented parallel to the `dim1` axis. Specify the location of the
    dip section with :obj:`distance` and :obj:`length` *or*
    :obj:`distance_idx` and :obj:`length` keyword parameters.

    .. plot::

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section(
        ...     'dip', dm.section.DipSection(distance=3000))
        >>> fig, ax = plt.subplots()
        >>> golfcube.quick_show('eta', idx=-1, ax=ax, ticks=True)
        >>> golfcube.sections['dip'].show_trace('r--', ax=ax)
        >>> plt.show()

    .. hint::

        Either :obj:`distance` *or* :obj:`distance_idx` must be specified.

    Parameters
    ----------
    *args : :obj:`DataCube` or `StratigraphyCube`
        The `Cube` object to link for underlying data. This option should be
        ommitted if using the :obj:`register_section` method of a `Cube`.

    distance : :obj:`float`, optional
        Distance *in `dim2` coordinates* from the `dim2` lower domain edge to
        place the section. The section location will be interpolated to the
        nearest grid cell. Mutually exclusive with `distance_idx`.

    distance_idx : :obj:`int`, optional
        Distance *in cell indices* from the `dim2` lower domain edge to place
        the section. Mutually exclusive with `distance`.

    length : :obj:`tuple` or :obj:`list` of `int` or `float`, optional
        A two-element tuple specifying the bounding points of the section in
        the `dim1` axis. Values are treated as cell indices
        if :obj:`distance_idx` is given and as `dim1` coordinates
        if :obj:`distance` is given. If no value is supplied, the section is
        drawn across the entire `dim1` axis (i.e., across the whole domain).
        Note that when indicies are given, the end point of length is treated
        as inclusive (e.g., `(10, 20)` results in a 21-cell section).

    x : :obj:`int`, optional, deprecated
        The number of cells in from the `dim2` lower domain edge. If used, the
        value is internally coerced to :obj:`distance_idx`.

    y : :obj:`int`, optional, deprecated
        The limits of the section. Defaults to the full `dim1`
        domain `length`. Specify as a two-element `tuple` or `list` of `int`,
        giving the lower and upper bounds of indices to span the section.

    **kwargs
        Keyword arguments are passed to `BaseSection.__init__()`. Supported
        options are `name`.

    Returns
    -------
    section : :obj:`DipSection`
        `DipSection` object with specified parameters. The section is
        automatically connected to the underlying `Cube` data source if the
        :obj:`register_section` method of a `Cube` is used to set up the
        section, or the `Cube` is passed as the first positional argument
        during instantiation.

    Examples
    --------

    Create a `DipSection` that is registered to a `DataCube` at specified
    `distance` in `dim2` coordinates, and spans the entire model domain:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section('dip', dm.section.DipSection(distance=3500))

        >>> # show the location and the "sandfrac" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.quick_show('eta', idx=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['dip'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['dip'].show('sandfrac', ax=ax[1])
        >>> plt.show()

    Create a `DipSection` that is registered to a `DataCube` at
    specified `distance_idx` index ``=75``, and spans the entire model domain:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section('dip75', dm.section.DipSection(distance_idx=75))

        >>> # show the location and the "sandfrac" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.quick_show('eta', idx=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['dip75'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['dip75'].show('sandfrac', ax=ax[1])
        >>> plt.show()

    Create a `DipSection` that is registered to a `StratigraphyCube` at
    specified `distance` in `dim2` coordinates, and spans only a range in the
    middle of the domain:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfstrat = dm.cube.StratigraphyCube.from_DataCube(golfcube, dz=0.1)
        >>> golfstrat.register_section(
        ...     'dip_part', dm.section.DipSection(distance=4000, length=(500, 1500)))

        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.quick_show('eta', idx=-1, ax=ax[0], ticks=True)
        >>> golfstrat.sections['dip_part'].show_trace('r--', ax=ax[0])
        >>> golfstrat.sections['dip_part'].show('velocity', ax=ax[1])
        >>> plt.show()
    """

    def __init__(
        self,
        *args,
        distance=None,
        distance_idx=None,
        length=None,
        x=None,
        y=None,
        **kwargs,
    ):
        # initialization is handled by the `LineSection` class
        # _compute_section_coords is called by the `BaseSection` class
        super().__init__(
            "dip",
            *args,
            distance=distance,
            distance_idx=distance_idx,
            length=length,
            x=x,
            y=y,
            **kwargs,
        )

    def _compute_section_coords(self):
        """Calculate coordinates of the dip section."""

        dim1_coords = self._underlying_dim1_coords
        dim2_coords = self._underlying_dim2_coords

        # if input length is None, we need to use endpoints of the dim1 coords
        if self._input_length is None:
            # if the value is given as distance
            if not (self._input_distance is None):
                _length = (
                    float(dim1_coords[0]),
                    float(dim1_coords[-1]),
                )
            # if the valueas idx
            else:
                _length = (0, self._L - 1)

        else:
            # quick check that value for length is valid
            if len(self._input_length) != 2:
                raise ValueError(
                    "Input `length` must be two element tuple or list, "
                    "but was {0}".format(str(self._input_length))
                )
            _length = self._input_length

        # if the value is given as distance
        if not (self._input_distance is None):
            # interpolate to an idx
            _idx = np.argmin(np.abs(np.array(dim2_coords) - self._input_distance))
            # treat length as coordinates
            #   should have some kind of checks here for valid values?
            _start_idx = np.argmin(np.abs(np.array(dim1_coords) - _length[0]))
            _end_idx = np.argmin(np.abs(np.array(dim1_coords) - _length[1]))
        else:
            # apply the input idx value
            _idx = int(self._input_distance_idx)
            # treat length as indices
            _start_idx, _end_idx = _length

        self._distance = float(dim2_coords[_idx])
        self._distance_idx = _idx
        # self._length = _length
        # self._length_idx = (_start_idx, _end_idx)

        # now compute the indices to use for the section
        self._dim1_idx = np.arange(_start_idx, _end_idx + 1, dtype=int)
        self._dim2_idx = np.tile(self._distance_idx, (len(self._dim1_idx)))
        self._length_idx = len(self._dim1_idx)

    @property
    def y(self):
        """Deprecated. Use :obj:`length_idx`."""
        warnings.warn("`.y` is a deprecated attribute. Use `.length_idx` instead.")
        return self._length_idx

    @property
    def x(self):
        """Deprecated. Use :obj:`distance_idx`.

        Start and end indices of section.
        """
        warnings.warn("`.x` is a deprecated attribute. Use `.distance_idx` instead.")
        return self._distance_idx


class CircularSection(BaseSection):
    """Circular section object.

    Section drawn as a circular cut, located a along the arc a specified
    `radius` from specified `origin`.  Specify the location of the circular
    section with `radius` and `origin` keyword parameter options.
    The circular section trace is interpolated to the nearest integer model
    domain cells, following the mid-point circle algorithm
    (:obj:`~deltametrics.utils.circle_to_cells`).

    .. plot::

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section(
        ...     'circular', dm.section.CircularSection(radius=1200))

        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots()
        >>> golfcube.quick_show('eta', idx=-1, ax=ax, ticks=True)
        >>> golfcube.sections['circular'].show_trace('r--', ax=ax)
        >>> plt.show()

    .. warning::

        This section will not work for unequal `dim1` and `dim2` coordinate
        spacing.

    Parameters
    ----------
    *args : :obj:`DataCube` or `StratigraphyCube`
        The `Cube` object to link for underlying data. This option should be
        ommitted if using the :obj:`register_section` method of a `Cube`.

    radius : :obj:`float`, optional
        The `radius` of the section in dimensional coordinates. This is the
        distance to locate the section from the :obj:`origin`. If no value is
        given, the `radius` defaults to half of the minimum model domain edge
        length.

    origin : :obj:`tuple` of `float`, optional
        The `origin` of the circular section in dimensional coordinates,
        specified as a two-element tuple ``(dim1, dim2)``. This is the center
        of the circle. If no value is given, the origin defaults to the
        center of the x-direction of the model domain, and offsets into the
        domain a distance of ``y == L0``, if this values can be determined.
        I.e., the origin defaults to be centered over the channel inlet.

    radius_idx : :obj:`float`, `int`, optional
        The `radius` of the section in cell indices. This is the distance to
        locate the section from the :obj:`origin`. Mutually exclusive
        with `radius`.

    origin_idx : :obj:`tuple` of `int`, optional
        The `origin` of the circular section in dimensional coordinates,
        specified as a two-element tuple ``(dim1, dim2)``. This is the center
        of the circle. Mutually exclusive with `origin`.

    **kwargs
        Keyword arguments are passed to `BaseSection.__init__()`. Supported
        options are `name`.

    Returns
    -------
    section : :obj:`CircularSection`
        `CircularSection` object with specified parameters. The section is
        automatically connected to the underlying `Cube` data source if the
        :obj:`register_section` method of a `Cube` is used to set up the
        section, or the `Cube` is passed as the first positional argument
        during instantiation.

    Examples
    --------

    Create a `CircularSection` that is registered to a `DataCube` with
    radius ``=1200``, and using the default `origin` options:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section(
        ...     'circular', dm.section.CircularSection(radius=1200))

        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.quick_show('eta', idx=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['circular'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['circular'].show('velocity', ax=ax[1])
        >>> plt.show()

    Create a `CircularSection` that is registered to a `StratigraphyCube` with
    radius index ``=50``, and the origin against the domain edge (using the
    `origin_idx` option):

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfstrat = dm.cube.StratigraphyCube.from_DataCube(golfcube, dz=0.1)
        >>> golfstrat.register_section(
        ...     'circular', dm.section.CircularSection(radius_idx=50,
        ...                                            origin_idx=(0, 100)))

        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.quick_show('eta', idx=-1, ax=ax[0], ticks=True)
        >>> golfstrat.sections['circular'].show_trace('r--', ax=ax[0])
        >>> golfstrat.sections['circular'].show('velocity', ax=ax[1])
        >>> plt.show()
    """

    def __init__(
        self,
        *args,
        radius=None,
        radius_idx=None,
        origin=None,
        origin_idx=None,
        **kwargs,
    ):

        self._origin = None
        self._radius = None

        # process the multiple possible arguments
        if (not (radius is None)) and (not (radius_idx is None)):
            raise ValueError("Cannot specify both `radius` and `radius_idx`.")
        if (not (origin is None)) and (not (origin_idx is None)):
            raise ValueError("Cannot specify both `origin` and `origin_idx`.")

        self._input_radius = radius
        self._input_origin = origin
        self._input_radius_idx = radius_idx
        self._input_origin_idx = origin_idx
        super().__init__("circular", *args, **kwargs)

    def _compute_section_coords(self):

        dim1_coords = self._underlying_dim1_coords
        dim2_coords = self._underlying_dim2_coords

        # determine the radius in indices
        if (self._input_radius is None) and (self._input_radius_idx is None):
            # if no inputs are provided, use a default based on domain dims
            if self._underlying_type == "cube":
                self._radius_idx = int(np.min(self._underlying.shape[1:]) / 2)
            else:
                self._radius_idx = int(np.min(self._underlying.shape) / 2)
        elif not (self._input_radius is None):
            # if radius was given in coords
            self._radius_idx = np.argmin(
                np.abs(np.array(dim1_coords) - self._input_radius)
            )
        else:
            # if radius was giveindices
            self._radius_idx = self._input_radius_idx

        # determine the origin in indices
        if (self._input_origin is None) and (self._input_origin_idx is None):
            # if no inputs are provided, try to guess from metadata or land
            if self._underlying_type == "cube":
                # cube is attached and can be used
                center_dim2 = int(
                    self._underlying.shape[2] // 2
                )  # center of the dim2 axis
                if self._underlying.meta is None:
                    # try and guess the value (should issue a warning?)
                    #   if no field called 'eta'?? this will fail.
                    warnings.warn(
                        "Trying to guess origin distance from dim1==0. "
                        "This is unlikely to work for data not generated from pyDeltaRCM."
                    )
                    land_width = np.minimum(
                        utils.guess_land_width_from_land(
                            self._underlying["eta"][-1, :, 0]
                        ),
                        5,
                    )
                else:
                    # extract L0 from the cube metadata
                    land_width = int(self._underlying.meta["L0"])
            else:
                # no cube is known
                land_width = 0
                center_dim2 = len(dim2_coords) // 2
            # combine into the origin as a (dim1, dim2) point
            self._origin_idx = (land_width, center_dim2)
        elif not (self._input_origin is None):
            # if origin was given in coords
            idx_dim1 = np.argmin(np.abs(np.array(dim1_coords) - self._input_origin[0]))
            idx_dim2 = np.argmin(np.abs(np.array(dim2_coords) - self._input_origin[1]))
            self._origin_idx = (idx_dim1, idx_dim2)
        else:
            # if origin was given in indices
            self._origin_idx = self._input_origin_idx

        # use the utility to compute the cells *in order*
        origin_idx_rev = tuple(reversed(self._origin_idx))  # input must be x,y
        xy = utils.circle_to_cells(origin_idx_rev, self._radius_idx)

        # store
        self._dim1_idx = xy[1]
        self._dim2_idx = xy[0]

        # store other variables
        self._radius = float(dim1_coords[self._radius_idx])
        self._origin = (
            float(dim1_coords[self._origin_idx[0]]),
            float(dim2_coords[self._origin_idx[1]]),
        )

    @property
    def radius(self):
        """Radius of the section in dimensional coordinates."""
        return self._radius

    @property
    def origin(self):
        """Origin of the section in dimensional coordinates.

        .. hint::

            Returned as a point ``(dim1, dim2)``, so will need to be reversed
            for plotting in Cartesian coordinates.
        """
        return self._origin


class RadialSection(BaseSection):
    """Radial section object.

    Section drawn as a radial cut, located a along the line starting from
    `origin` and proceeding away in direction specified by azimuth. Specify
    the location of the radial section with `azimuth` and `origin`
    keyword parameter options. The radial section trace is interpolated to the
    nearest integer model domain cells, following the a line-walking algorithm
    (:obj:`~deltametrics.utils.line_to_cells`).

    .. plot::

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section(
        ...     'radial', dm.section.RadialSection(azimuth=65))

        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots()
        >>> golfcube.quick_show('eta', idx=-1, ax=ax, ticks=True)
        >>> golfcube.sections['radial'].show_trace('r--', ax=ax)
        >>> plt.show()

    .. important::

        The `origin` attempts to detect the land width from bed elevation
        changes, but should use the value of ``L0`` recorded in the netcdf
        file, or defined in the cube.

    .. important::

        This Section type will only work for deltas with an inlet along the
        ``dim1`` lower domain edge. For other delta configurations, specify a
        radial section by defining two end points and instantiating a
        `Section` with the :obj:`PathSection`. A patch for this is welcomed!

    Parameters
    ----------
    *args : :obj:`DataCube` or `StratigraphyCube`
        The `Cube` object to link for underlying data. This option should be
        ommitted if using the :obj:`register_section` method of a `Cube`.

    azimuth : :obj:`float`, `int`, optional
        The `azimuth` of the section, directed away from the origin. If no
        value is given, the `azimuth` defaults to ``90``.

    origin : :obj:`tuple` of `float`, optional
        The `origin` of the radial section in dimensional coordinates,
        specified as a two-element tuple ``(dim1, dim2)``. This is the
        starting point of the radial line. If no value is given, the origin
        defaults to the center of the x-direction of the model domain, and
        offsets into the domain a distance of ``y == L0``, if this values can
        be determined. I.e., the origin defaults to be centered over the
        channel inlet.

    origin_idx : :obj:`tuple` of `int`, optional
        The `origin` of the radial section in dimensional coordinates,
        specified as a two-element tuple ``(dim1, dim2)``. This is the
        starting point of the radial line. Mutually exclusive
        with `origin`.

    length : :obj:`float`, `int`, optional
        The length of the section, assumed to be in the same coordinate
        specification as the origin was defined. If no value is given, the
        length defaults to the length required to reach a domain boundary
        (if a connection to underlying `Cube` exists). If
        neither :obj:`origin` or :obj:`origin_idx` is specified, `length` is
        assumed to be in dimensional coordinates.

        .. important::

            length is used as a *guide* for the section length, as the section
            is approximated to cell indices. This is unlikely to work as
            expected for grid spacing that is not equal in both dimensions.

    **kwargs
        Keyword arguments are passed to `BaseSection.__init__()`. Supported
        options are `name`.

    Returns
    -------
    section : :obj:`RadialSection`
        `RadialSection` object with specified parameters. The section is
        automatically connected to the underlying `Cube` data source if the
        :obj:`register_section` method of a `Cube` is used to set up the
        section, or the `Cube` is passed as the first positional argument
        during instantiation.

    Examples
    --------

    Create a `RadialSection` that is registered to a `DataCube` at
    specified `origin` coordinate, and spans the entire domain:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section(
        ...     'radial', dm.section.RadialSection(azimuth=45))

        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.quick_show('eta', idx=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['radial'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['radial'].show('velocity', ax=ax[1])
        >>> plt.show()

    Create several `RadialSection` objects, spaced out across the domain,
    connected to a `StratigraphyCube`. Each section should be shorter than
    the full domain width.

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfstrat = dm.cube.StratigraphyCube.from_DataCube(golfcube, dz=0.1)

        >>> fig, ax = plt.subplots(2, 3, figsize=(9, 3))
        >>> ax = ax.flatten()
        >>> golfcube.quick_show('eta', idx=-1, ax=ax[1], ticks=True)
        >>> ax[1].tick_params(labelsize=8)

        >>> azims = np.linspace(0, 180, num=5)
        >>> idxs = [2, 5, 4, 3, 0]  # indices in the order to draw to match 0-->180
        >>> for i, (azim, idx) in enumerate(zip(azims, idxs)):
        ...     sec = dm.section.RadialSection(golfstrat, azimuth=azim, length=4000)
        ...     sec.show_trace('r--', ax=ax[1])
        ...     sec.show('time', ax=ax[idx], colorbar=False)
        ...     ax[idx].text(3000, 0, 'azimuth: {0}'.format(azim), ha='center', fontsize=8)
        ...     ax[idx].tick_params(labelsize=8)
        ...     ax[idx].set_ylim(-4, 1)

        >>> plt.show()
    """

    # @staticmethod
    # def _layout(shape, origin_idx, length_idx, )

    def __init__(
        self, *args, azimuth=None, origin=None, origin_idx=None, length=None, **kwargs
    ):

        self._azimuth = None
        self._origin = None

        # process the multiple possible arguments
        if (not (origin is None)) and (not (origin_idx is None)):
            raise ValueError("Cannot specify both `origin` and `origin_idx`.")

        self._input_azimuth = azimuth
        self._input_origin = origin
        self._input_length = length
        self._input_origin_idx = origin_idx

        super().__init__("radial", *args, **kwargs)

    def _determine_layout(self):
        # find the coordinates
        pass

    def _compute_section_coords(self):
        """Compute the coordinates of the section.

        This method is called by `connect`, and should generally not be used
        directly by the user, unless some advanced use case requires.
        """
        _L = self._L
        _W = self._W
        dim1_coords = self._underlying_dim1_coords
        dim2_coords = self._underlying_dim2_coords

        # determine the azimuth
        if self._input_azimuth is None:
            self._azimuth = 90
        else:
            self._azimuth = self._input_azimuth

        # determine the origin in indices
        if (self._input_origin is None) and (self._input_origin_idx is None):
            # if no inputs are provided, try to guess from metadata or land,
            #   or fall back on just using (0, len(dim2) / 2)
            if self._underlying_type == "cube":
                # cube is attached and can be used
                center_dim2 = int(
                    self._underlying.shape[2] // 2
                )  # center of the dim2 axis
                if self._underlying.meta is None:
                    # try and guess the value (should issue a warning?)
                    #   if no field called 'eta'?? this will fail.
                    warnings.warn(
                        "Trying to guess origin distance from dim1==0. "
                        "This is unlikely to work for data not generated from pyDeltaRCM."
                    )
                    land_width = np.minimum(
                        utils.guess_land_width_from_land(
                            self._underlying["eta"][-1, :, 0]
                        ),
                        5,
                    )
                else:
                    # extract L0 from the cube metadata
                    land_width = int(self._underlying.meta["L0"])
            else:
                # no cube is known
                land_width = 0
                center_dim2 = len(dim2_coords) // 2
            # combine into the origin as a (dim1, dim2) point
            self._origin_idx = (land_width, center_dim2)

        elif not (self._input_origin is None):
            idx_dim1 = np.argmin(np.abs(np.array(dim1_coords) - self._input_origin[0]))
            idx_dim2 = np.argmin(np.abs(np.array(dim2_coords) - self._input_origin[1]))
            self._origin_idx = (idx_dim1, idx_dim2)

        else:
            # if origin was given in indices
            self._origin_idx = self._input_origin_idx

        # determine the length of the line to travel
        #   first, find the line function of the azimuth
        theta = self.azimuth
        m = np.tan(theta * np.pi / 180)
        b = self._origin_idx[0] - m * self._origin_idx[1]

        # use vector math to determine end point len along azimuth
        #   vector is from (0, b) to (origin)
        if self.azimuth <= 90.0 and self.azimuth >= 0:
            vec = np.array([self._origin_idx[1] - 0, self._origin_idx[0] - b])
        elif self.azimuth > 90 and self.azimuth <= 180:
            vec = np.array([0 - self._origin_idx[1], b - self._origin_idx[0]])
        else:
            raise ValueError("Azimuth must be in range (0, 180).")
            # note, this logic should be able to be extended to handle
            #  cases where >180 is needed.

        # need to determine the length to travel
        if self._input_length is None:
            # if no input, find the intersection with an edge
            if self.azimuth <= 90.0 and self.azimuth >= 0:
                # find the intersection of two bounding lines
                length_to_W = (
                    (_W - self._origin_idx[1]) ** 2
                    + ((m * _W + b) - self._origin_idx[0]) ** 2
                ) ** 0.5
                length_to_L = (
                    (((_L - b) / (m + 1e-16)) - self._origin_idx[1]) ** 2
                    + (_L - self._origin_idx[0]) ** 2
                ) ** 0.5
                _length_idx = np.minimum(length_to_W, length_to_L)
            elif self.azimuth > 90 and self.azimuth <= 180:
                length_to_0 = (
                    (self._origin_idx[1] - 0) ** 2
                    + (self._origin_idx[0] - (m * (0) + b)) ** 2
                ) ** 0.5
                length_to_L = (
                    (((_L - b) / (m + 1e-16)) - self._origin_idx[1]) ** 2
                    + (_L - self._origin_idx[0]) ** 2
                ) ** 0.5
                _length_idx = np.minimum(length_to_0, length_to_L)
            else:
                raise ValueError("Azimuth must be in range (0, 180).")
                # note, this logic should be able to be extended to handle
                #  cases where >180 is needed.
        else:
            # if input length given
            underlying_dx = float(dim1_coords[1] - dim1_coords[0])
            if not (self._input_origin is None):
                # if the origin was given as dimensional coords
                _length_idx = self._input_length // underlying_dx
            elif not (self._input_origin_idx is None):
                # if the origin as
                _length_idx = self._input_length
            else:
                # interpret the length value as coordinates
                _length_idx = self._input_length // underlying_dx

        # we have (all in indicies) origin, length, and vector
        vec_norm = vec / np.sqrt(vec[1] ** 2 + vec[0] ** 2)
        end_point = (
            int(self._origin_idx[1] + _length_idx * vec_norm[0]),
            int(self._origin_idx[0] + _length_idx * vec_norm[1]),
        )

        # note that origin idx and end point are in x-y cartesian convention!
        origin_idx_rev = tuple(reversed(self._origin_idx))  # input must be x,y
        x, y = utils.line_to_cells(origin_idx_rev, end_point)

        # validate and clean the xy array
        yvalid = np.logical_and(y >= 0, y <= (_L - 1))
        xvalid = np.logical_and(x >= 0, x <= (_W - 1))
        xyvalid = np.logical_and(yvalid, xvalid)

        self._dim1_idx = y[xyvalid]
        self._dim2_idx = x[xyvalid]

        self._end_point_idx = (self._dim1_idx, self._dim2_idx)
        self._origin = (
            float(dim1_coords[self._origin_idx[0]]),
            float(dim2_coords[self._origin_idx[1]]),
        )

    @property
    def azimuth(self):
        """Azimuth of section (degrees)."""
        return self._azimuth

    @property
    def origin(self):
        """Origin of the section in dimensional coordinates.

        .. hint::

            Returned as a point ``(dim1, dim2)``, so will need to be reversed
            for plotting in Cartesian coordinates.
        """
        return self._origin
