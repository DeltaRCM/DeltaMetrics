import abc
import warnings

import numpy as np
from scipy import sparse
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from . import cube
from . import plot
from . import utils


@xr.register_dataarray_accessor("strat")
class StratigraphicInformation:
    """Stratigraphic information accessor for SectionVariables.

    Provides an `xarray` accessor called "strat" for holding stratigraphic
    information, and enabling computations and visualizations that depend on
    stratigraphic preservation information.
    """
    _spacetime_names = ['full', 'spacetime', 'as spacetime', 'as_spacetime']
    _preserved_names = ['psvd', 'preserved', 'as preserved', 'as_preserved']
    _stratigraphy_names = ['strat', 'strata', 'stratigraphy',
                           'as stratigraphy', 'as_stratigraphy']

    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        self._knows_stratigraphy = xarray_obj.knows_stratigraphy
        self._knows_spacetime = xarray_obj.knows_spacetime

    def add_information(self, _psvd_mask=None, _strat_attr=None):
        # check information is valid for object
        if (_psvd_mask is not None):
            _psvd_mask = np.asarray(_psvd_mask)
            if _psvd_mask.shape != self._obj.shape:
                raise ValueError(
                    'Shape of "_psvd_mask" incompatible with "_data" array.')
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
            raise AttributeError(
                'No "spacetime" or "preserved" information available.')
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
            _psvd_data = self._obj.data[self.strat_attr['psvd_idx']]
            _sp = sparse.coo_matrix((_psvd_data,
                                     (self.strat_attr['z_sp'],
                                      self.strat_attr['s_sp'])))
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
        CubeInstance : :obj:`~deltametrics.cube.BaseCube` subclass, optional
            Connect to this cube. No connection is made if cube is not
            provided.

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
        self.cube = None

        self.section_type = section_type
        self._name = name

        if len(args) > 1:
            raise ValueError('Expected single positional argument to \
                             %s instantiation.'
                             % type(self))

        if len(args) > 0:
            self.connect(args[0])
        else:
            pass

    def connect(self, CubeInstance, name=None):
        """Connect this Section instance to a Cube instance.
        """
        if not issubclass(type(CubeInstance), cube.BaseCube):
            raise TypeError('Expected type is subclass of {_exptype}, '
                            'but received was {_gottype}.'.format(
                                _exptype=type(cube.BaseCube),
                                _gottype=type(CubeInstance)))
        self.cube = CubeInstance
        self._variables = self.cube.variables
        self.name = name  # use the setter to determine the _name
        self._compute_section_coords()
        self._compute_section_attrs()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, var):
        if (self._name is None):
            # _name is not yet set
            self._name = var or self.section_type
        else:
            # _name is already set
            if not (var is None):
                warnings.warn(
                    UserWarning("`name` argument supplied to instantiated "
                                "`Section` object. To change the name of "
                                "a Section, you must set the attribute "
                                "directly with `section._name = 'name'`."))
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
        self._idx_trace = np.column_stack((self._dim1_idx, self._dim2_idx))
        self._trace = np.column_stack((self.cube._dim2_idx[self._dim2_idx],
                                       self.cube._dim1_idx[self._dim1_idx]))
        
        # compute along section distance and place into a DataArray
        _s = np.cumsum(np.hstack(
            (0, np.sqrt((self._trace[1:, 0] - self._trace[:-1, 0])**2
             + (self._trace[1:, 1] - self._trace[:-1, 1])**2))))
        self._s = xr.DataArray(_s, name='s', dims=['s'], coords={'s': _s})

        # take z from the underlying cube, should be a DataArray
        self._z = self.cube.z

        # set shape from the coordinates
        self._shape = (len(self._z), len(self._s))

    @property
    def idx_trace(self):
        """Indices of section points in the dim1, dim2 plane.
        """
        return self._idx_trace

    @property
    def trace(self):
        """Coordinates of the section in the x-y plane.

        .. note:: stack of [dim2, dim1].
        """
        return self._trace

    @property
    def s(self):
        """Along-section coordinate."""
        return self._s

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
        """List of variables.
        """
        return self._variables

    @property
    def strat_attr(self):
        """Stratigraphic attributes data object.

        Raises
        ------
        NoStratigraphyError
            If no stratigraphy information is found for the section.
        """
        if self.cube._knows_stratigraphy:
            return self.cube.strat_attr
        else:
            raise utils.NoStratigraphyError(obj=self, var='strat_attr')

    def __getitem__(self, var):
        """Get a slice of the section.

        Slicing the section instance creates a
        :obj:`~deltametrics.section.SectionVariable` instance from data for
        variable ``var``.

        .. note:: We only support slicing by string.

        Parameters
        ----------
        var : :obj:`str`
            Which variable to slice.

        Returns
        -------
        SectionVariable : :obj:`~deltametrics.section.SectionVariable` instance
            SectionVariable instance for variable ``var``.
        """
        if isinstance(self.cube, cube.DataCube):
            if self.cube._knows_stratigraphy:
                _xrDA = xr.DataArray(
                    self.cube[var].data[:, self._dim1_idx, self._dim2_idx],
                    coords={"s": self._s, self._z.dims[0]: self._z},
                    dims=[self._z.dims[0], 's'],
                    name=var,
                    attrs={'slicetype': 'data_section',
                           'knows_stratigraphy': True,
                           'knows_spacetime': True})
                _xrDA.strat.add_information(
                    _psvd_mask=self.cube.strat_attr.psvd_idx[:, self._dim1_idx, self._dim2_idx],  # noqa: E501
                    _strat_attr=self.cube.strat_attr(
                        'section', self._dim1_idx, self._dim2_idx))
                return _xrDA
            else:
                _xrDA = xr.DataArray(
                    self.cube[var].data[:, self._dim1_idx, self._dim2_idx],
                    coords={"s": self._s, self._z.dims[0]: self._z},
                    dims=[self._z.dims[0], 's'],
                    name=var,
                    attrs={'slicetype': 'data_section',
                           'knows_stratigraphy': False,
                           'knows_spacetime': True})
                return _xrDA
        elif isinstance(self.cube, cube.StratigraphyCube):
            _xrDA = xr.DataArray(
                    self.cube[var].data[:, self._dim1_idx, self._dim2_idx],
                    coords={"s": self._s, self._z.dims[0]: self._z},
                    dims=[self._z.dims[0], 's'],
                    name=var,
                    attrs={'slicetype': 'stratigraphy_section',
                           'knows_stratigraphy': True,
                           'knows_spacetime': False})
            return _xrDA
        elif (self.cube is None):
            raise AttributeError(
                'No cube connected. Are you sure you ran `.connect()`?')
        else:
            raise TypeError('Unknown Cube type encountered: %s'
                            % type(self.cube))

    def show(self, SectionAttribute, style='shaded', data=None,
             label=False, colorbar=True, colorbar_label=False, ax=None):
        """Show the section.

        Method enumerates convenient routines for visualizing sections of data
        and stratigraphy. Includes support for multiple data `style` and
        mutuple `data` choices as well.

        .. note::

            The colors for `style='lines'` are determined from the left-end
            edge node, and colors for the `style='shaded'` mesh are determined
            from the lower-left-end edge node of the quad.

        Parameters
        ----------

        SectionAttribute : :obj:`str`, :obj:`SectionVariableInstance`
            Which attribute to show. Can be a string for a named `Cube`
            attribute, or any arbitrary data.

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
            ...     'demo', dm.section.StrikeSection(y=5))
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
            ...     'demo', dm.section.StrikeSection(y=5))

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
        # process arguments and inputs
        if not ax:
            ax = plt.gca()
        _varinfo = self.cube.varset[SectionAttribute] if \
            issubclass(type(self.cube), cube.BaseCube) else \
            plot.VariableSet()[SectionAttribute]
        SectionVariableInstance = self[SectionAttribute]

        # main routines for plot styles
        if style in ['shade', 'shaded']:
            _data, _X, _Y = plot.get_display_arrays(SectionVariableInstance,
                                                    data=data)
            ci = ax.pcolormesh(_X, _Y, _data, cmap=_varinfo.cmap,
                               norm=_varinfo.norm,
                               vmin=_varinfo.vmin, vmax=_varinfo.vmax,
                               rasterized=True, shading='auto')
        elif style in ['line', 'lines']:
            _data, _segments = plot.get_display_lines(SectionVariableInstance,
                                                      data=data)
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
                _colorbar_label = _varinfo.label if (colorbar_label is True) \
                    else str(colorbar_label)  # use custom if passed
                cb.ax.set_ylabel(_colorbar_label, rotation=-90, va="bottom")
        ax.margins(y=0.2)
        if label:
            _label = _varinfo.label if (label is True) else str(
                label)  # use custom if passed
            ax.text(0.99, 0.8, _label, fontsize=10,
                    horizontalalignment='right', verticalalignment='center',
                    transform=ax.transAxes)
        
        # set the limits of the plot accordingly
        xmin, xmax, ymin, ymax = plot.get_display_limits(
            SectionVariableInstance, data=data)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

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

        _label = kwargs.pop('label', self.name)

        _x = self.cube._dim2_idx[self._dim2_idx]
        _y = self.cube._dim1_idx[self._dim1_idx]

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

    .. important::

        The vertex coordinates must be specified as cell indices along `dim1`
        and `dim2`. That is to sat *not* actual `dim0` and `dim1` coordinate
        values *and not* x-y cartesian indices (actually y-x indices).
        Specifying coordinates is is a needed patch.

    Parameters
    ----------
    *args : :obj:`DataCube` or `StratigraphyCube`
        The `Cube` object to link for underlying data. This option should be
        ommitted if using the :obj:`register_section` method of a `Cube`.

    path : :obj:`ndarray`
        An `(N, 2)` `ndarray` specifying the `dim1-dim2` pairs of coordinates
        that define the verticies of the path to extract the section from.

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

    To create a `PathSection` that is registered to a `DataCube` at
    specified coordinates:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section('path', dm.section.PathSection(
        ...     path=np.array([[3, 50], [17, 65], [10, 130]])))
        >>>
        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['path'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['path'].show('velocity', ax=ax[1])
        >>> plt.show()
    """

    def __init__(self, *args, path, **kwargs):
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
        self._input_path = path
        super().__init__('path', *args, **kwargs)

    def _compute_section_coords(self):
        """Calculate coordinates of the strike section.
        """
        # convert the points into segments into lists of cells
        # breakpoint()
        _segs = utils.coordinates_to_segments(np.fliplr(self._input_path))
        _cell = utils.segments_to_cells(_segs)

        # determine only unique coordinates along the path
        self._path = np.unique(_cell, axis=0)
        self._vertices = np.unique(self._input_path, axis=0)

        self._x = self._path[:, 0]
        self._y = self._path[:, 1]
        self._dim1_idx = self._path[:, 1]
        self._dim2_idx = self._path[:, 0]

    @property
    def path(self):
        """Path of the PathSection.

        Returns same as `trace` property.
        """
        return self.trace


class StrikeSection(BaseSection):
    """Strike section object.

    Section oriented along the delta strike (i.e., perpendicular to an inlet
    channel). Specify the location of the strike section with :obj:`y` and
    :obj:`x` keyword parameter options.

    .. important::

        The `y` and `x` parameters must be specified as cell indices (not
        actual x and y coordinate values). This is a needed patch.

    Parameters
    ----------
    *args : :obj:`DataCube` or `StratigraphyCube`
        The `Cube` object to link for underlying data. This option should be
        ommitted if using the :obj:`register_section` method of a `Cube`.

    y : :obj:`int`, optional
        The `y` location of the section. This is the distance to locate the
        section from the domain edge with a channel inlet. Defaults to ``0``
        if no value is given.

    x : :obj:`int`, optional
        The `x` limits for the section. Defaults to the full domain width.
        Specify as a two-element `tuple` or `list` of `int`, giving the lower
        and upper bounds of `x` values to span the section.

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

    To create a `StrikeSection` that is registered to a `DataCube` at
    specified `y` coordinate ``=10``, and spans the entire model domain:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section('strike', dm.section.StrikeSection(y=10))
        >>>
        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['strike'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['strike'].show('velocity', ax=ax[1])
        >>> plt.show()

    Similarly, create a `StrikeSection` that is registered to a
    `StratigraphyCube` at specified `y` coordinate ``=20``, and spans only the
    left side of the model domain:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> sc8cube = dm.cube.StratigraphyCube.from_DataCube(golfcube)
        >>> sc8cube.register_section(
        ...     'strike_half', dm.section.StrikeSection(y=20, x=[0, 120]))

        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
        >>> sc8cube.sections['strike_half'].show_trace('r--', ax=ax[0])
        >>> sc8cube.sections['strike_half'].show('velocity', ax=ax[1])
        >>> plt.show()
    """

    def __init__(self, *args, y=None, x=None, **kwargs):

        self.y = y  # strike coord scalar
        self._input_xlim = x  # the input x lims
        super().__init__('strike', *args, **kwargs)

    def _compute_section_coords(self):
        """Calculate coordinates of the strike section.
        """
        if self._input_xlim is None:
            _nx = self.cube['eta'].shape[2]
            self._dim2_idx = np.arange(_nx)
        else:
            self._dim2_idx = np.arange(self._input_xlim[0], self._input_xlim[1])
            _nx = len(self._dim2_idx)
        self._dim1_idx = np.tile(self.y, (_nx))


class DipSection(BaseSection):
    """Dip section object.

    Section oriented along the delta dip (i.e., parallel to inlet channel).
    Specify the location of the dip section with :obj:`x` and :obj:`y`
    keyword parameter options.

    .. important::

        The `y` and `x` parameters must be specified as cell indices (not
        actual x and y coordinate values). This is a needed patch.

    Parameters
    ----------
    *args : :obj:`DataCube` or `StratigraphyCube`
        The `Cube` object to link for underlying data. This option should be
        ommitted if using the :obj:`register_section` method of a `Cube`.

    x : :obj:`int`, optional
        The `x` location of the section. This is the distance to locate the
        section from the domain edge with a channel inlet. Defaults to ``-1``
        if no value is given, which centers the section along the domain.

    y : :obj:`int`, optional
        The `y` limits for the section. Defaults to the full domain width.
        Specify as a two-element `tuple` or `list` of `int`, giving the lower
        and upper bounds of `y` values to span the section.

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

    To create a `DipSection` that is registered to a `DataCube` at
    specified `x` coordinate ``=130``, and spans the entire model domain:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section('dip', dm.section.DipSection(x=130))
        >>>
        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['dip'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['dip'].show('velocity', ax=ax[1])
        >>> plt.show()

    Similarly, create a `DipSection` that is registered to a
    `StratigraphyCube` at the inlet, which spans only the
    first 50 cells of the model domain:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> sc8cube = dm.cube.StratigraphyCube.from_DataCube(golfcube)
        >>> sc8cube.register_section(
        ...     'dip_short', dm.section.DipSection(y=[0, 50]))

        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
        >>> sc8cube.sections['dip_short'].show_trace('r--', ax=ax[0])
        >>> sc8cube.sections['dip_short'].show('velocity', ax=ax[1])
        >>> plt.show()
    """

    def __init__(self, *args, x=-1, y=None, **kwargs):

        self.x = x  # dip coordinate scalar
        self._input_ylim = y  # input y lims
        super().__init__('dip', *args, **kwargs)

    def _compute_section_coords(self):
        """Calculate coordinates of the dip section."""
        # if x is -1 pick center cell
        if self.x == -1:
            self.x = int(self.cube['eta'].shape[2] / 2)

        if self._input_ylim is None:
            _ny = self.cube['eta'].shape[1]
            self._dim1_idx = np.arange(_ny)
        else:
            self._dim1_idx = np.arange(self._input_ylim[0], self._input_ylim[1])
            _ny = len(self._dim1_idx)
        self._dim2_idx = np.tile(self.x, (_ny))


class CircularSection(BaseSection):
    """Circular section object.

    Section drawn as a circular cut, located a along the arc a specified
    radius from specified origin.  Specify the location of the circular section
    with :obj`radius` and :obj:`origin` keyword parameter options. The
    circular section trace is interpolated to the nearest integer model domain
    cells, following the mid-point circle algorithm
    (:obj:`~deltametrics.utils.circle_to_cells`).

    .. important::

        The `radius` and `origin` parameters must be specified as cell indices
        (not actual x and y coordinate values). This is a needed patch.

    .. important::

        The `origin` attempts to detect the land width from bed elevation
        changes, but should use the value of ``L0`` recorded in the netcdf
        file, or defined in the cube.

    Parameters
    ----------
    *args : :obj:`DataCube` or `StratigraphyCube`
        The `Cube` object to link for underlying data. This option should be
        ommitted if using the :obj:`register_section` method of a `Cube`.

    radius : :obj:`float`, `int`, optional
        The `radius` of the section. This is the distance to locate the
        section from the :obj:`origin`. If no value is given, the `radius`
        defaults to half of the minimum model domain edge length if it can be
        determined, otherwise defaults to ``1``.

    origin : :obj:`tuple` or `list` of `int`, optional
        The `origin` of the circular section. This is the center of the
        circle. If no value is given, the origin defaults to the center of the
        x-direction of the model domain, and offsets into the domain a
        distance of ``y == L0``, if these values can be determined. I.e., the
        origin defaults to be centered over the channel inlet. If no value is
        given, and these values cannot be determined, the origin defaults to
        ``(0, 0)``.

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

    To create a `CircularSection` that is registered to a `DataCube` with
    radius ``=30``, and using the default `origin` options:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section(
        ...     'circular', dm.section.CircularSection(radius=30))

        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['circular'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['circular'].show('velocity', ax=ax[1])
        >>> plt.show()
    """

    def __init__(self, *args, radius=None, origin=None, **kwargs):

        self._input_radius = radius
        self._input_origin = origin
        super().__init__('circular', *args, **kwargs)

    def _compute_section_coords(self):
        if (self._input_radius is None):
            self.radius = int(np.min(self.cube.shape[1:]) / 2)
        else:
            self.radius = self._input_radius

        if (self._input_origin is None):
            if (self.cube.meta is None):
                # try and guess the value (should issue a warning?)
                land_width = np.minimum(utils.guess_land_width_from_land(
                    self.cube['eta'][-1, :, 0]), 5)
            else:
                # extract L0 from the cube
                land_width = self.cube.meta['L0']
            self.origin = (int(self.cube.shape[2] / 2),
                           land_width)
        else:
            self.origin = self._input_origin

        xy = utils.circle_to_cells(self.origin, self.radius)

        self._dim1_idx = xy[1]
        self._dim2_idx = xy[0]


class RadialSection(BaseSection):
    """Radial section object.

    Section drawn as a radial cut, located a along the line starting from
    `origin` and proceeding away in direction specified by azimuth. Specify
    the location of the radial section with :obj`azimuth` and :obj:`origin`
    keyword parameter options. The radial section trace is interpolated to the
    nearest integer model domain cells, following the a line-walking algorithm
    (:obj:`~deltametrics.utils.line_to_cells`).

    .. important::

        The `origin` parameter must be specified as cell indices (not actual x
        and y coordinate values). This is a needed patch.

    .. important::

        The `origin` attempts to detect the land width from bed elevation
        changes, but should use the value of ``L0`` recorded in the netcdf
        file, or defined in the cube.

    .. important::

        This Section type will only work for deltas with an inlet along the
        ``y=0`` line. For other delta configurations, specify a radial
        section by defining two end points and instantiating a `Section` with
        the :obj:`PathSection`.

    Parameters
    ----------
    *args : :obj:`DataCube` or `StratigraphyCube`
        The `Cube` object to link for underlying data. This option should be
        ommitted if using the :obj:`register_section` method of a `Cube`.

    azimuth : :obj:`float`, `int`, optional
        The `azimuth` of the section, directed away from the origin. If no
        value is given, the `azimuth` defaults to ``90``.

    origin : :obj:`tuple` or `list` of `int`, optional
        The `origin` of the radial section. This is the "start" of the radial
        line. If no value is given, the origin defaults to the center of the
        x-direction of the model domain, and offsets into the domain a
        distance of ``y == L0``, if these values can be determined. I.e., the
        origin defaults to be centered over the channel inlet. If no value is
        given and these values cannot be determined, the origin defaults to
        ``(0, 0)``.

    length : :obj:`float`, `int`, optional
        The length of the section (note this must be given in pixel length).
        If no value is given, the length defaults to the length required to
        reach a model boundary (if a connection to underlying `Cube` exists).
        Otherwise, length is set to ``1``.

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

    To create a `RadialSection` that is registered to a `DataCube` at
    specified `origin` coordinate, and spans the entire model domain:

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> golfcube.register_section(
        ...     'radial', dm.section.RadialSection(azimuth=45))

        >>> # show the location and the "velocity" variable
        >>> fig, ax = plt.subplots(2, 1, figsize=(8, 4))
        >>> golfcube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
        >>> golfcube.sections['radial'].show_trace('r--', ax=ax[0])
        >>> golfcube.sections['radial'].show('velocity', ax=ax[1])
        >>> plt.show()
    """
    def __init__(self, *args, azimuth=None, origin=None, length=None,
                 **kwargs):
        self._input_azimuth = azimuth
        self._input_origin = origin
        self._input_length = length
        super().__init__('radial', *args, **kwargs)

    def _compute_section_coords(self):

        # determine the azimuth
        if (self._input_azimuth is None):
            self.azimuth = 90
        else:
            self.azimuth = self._input_azimuth

        # determine the origin of the line
        if (self._input_origin is None):
            if (self.cube.meta is None):
                # try and guess the value (should issue a warning?)
                land_width = np.minimum(utils.guess_land_width_from_land(
                    self.cube['eta'][-1, :, 0]), 5)
            else:
                # extract L0 from the cube
                land_width = self.cube.meta['L0']
            self.origin = (int(self.cube.shape[2] / 2),
                           land_width)
        else:
            self.origin = self._input_origin

        # determine the length of the line to travel
        # find the line of the azimuth
        theta = self.azimuth
        m = np.tan(theta * np.pi / 180)
        b = self.origin[1] - m * self.origin[0]
        if (self._input_length is None):
            # if no input
            # find the intersection with an edge
            if self.azimuth <= 90.0 and self.azimuth >= 0:
                dx = (self.cube.W - self.origin[0])
                dy = (np.tan(theta * np.pi / 180) * dx)
                if dy <= self.cube.L:
                    end_y = int(np.minimum(
                        m * (self.cube.W) + b, self.cube.L - 1))
                    end_point = (self.cube.W - 1, end_y)
                else:
                    end_x = int(np.minimum(
                        (self.cube.L - b) / m, self.cube.W - 1))
                    end_point = (end_x, self.cube.L - 1)
            elif self.azimuth > 90 and self.azimuth <= 180:
                dx = (self.origin[0])
                dy = (np.tan(theta * np.pi / 180) * dx)
                if np.abs(dy) <= self.cube.L:
                    end_y = b
                    end_point = (0, end_y)
                else:
                    end_x = int(np.maximum((self.cube.L - b) / m,
                                           0))
                    end_point = (end_x, self.cube.L - 1)
            else:
                raise ValueError('Azimuth must be in range (0, 180).')
        else:
            # if input length
            _len = self._input_length
            # use vector math to determine end point len along azimuth
            #   vector is from (0, b) to (origin)
            vec = np.array([self.origin[0] - 0, self.origin[1] - b])
            vec_norm = vec / np.sqrt(vec**2)
            end_point = (self.origin[0] + _len*vec_norm[0],
                         self.origin[1] + _len*vec_norm[1])

        xy = utils.line_to_cells(self.origin, end_point)

        self._dim1_idx = xy[1]
        self._dim2_idx = xy[0]
