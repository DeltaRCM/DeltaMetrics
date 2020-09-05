import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as ptch
import matplotlib.collections as coll
import mpl_toolkits.axes_grid1 as axtk

from . import io
from . import strat
from . import section

# plotting utilities


class VariableInfo(object):
    """Variable styling and information.

    This class holds information for a specific underlying data variable
    (e.g., ``eta`` or ``velocity``). Properties are used throughout
    DeltaMetrics for plotting.

    Examples
    --------

    You can override any of the parameters by passing a value during
    instantiation of the VariableInfo. For example, to set the colormap
    for a variable ``vegetation_density`` as a range of green shades we
    would create the VariableInfo as:

    .. doctest::

        >>> from deltametrics.plot import VariableInfo
        >>> veg = VariableInfo('vegetation_density',
        ...                    cmap='Greens')

        >>> veg.cmap
        <matplotlib.colors.LinearSegmentedColormap object at 0x...>

        >>> veg.cmap.N
        64

    which creates a 64 interval colormap from the matplotlib `Greens`
    colormap. If instead we wanted just three colors, we could specify the
    colormap manually:

    .. doctest::

        >>> veg3 = VariableInfo('vegetation_density',
        ...                     cmap=plt.cm.get_cmap('Greens', 3))
        >>> veg3.cmap.N
        3

    We can then set a more human-readable label for the variable:

    .. doctest::

        >>> veg3.label = 'vegetation density'

    """

    def __init__(self, name, **kwargs):
        """Initialize the VariableInfo.

        Parameters
        ----------
        name : :obj:`str`
            Name for variable to access from.

        cmap : :obj:`matplotlib.colors.Colormap`, :obj:`str`, optional
            Colormap to use to diplay the variable.

        label : :obj:`str`, optional
            How to display the variable name when plotting (i.e., a
            human-readable name). If not specified, the value of `name` is
            used.

        norm : :obj:`matplotlib.colors.Norm`, optional
            ???

        vmin : :obj:`float`, optional
            Limit the colormap or axes to this lower-bound value.

        vmax : :obj:`float`, optional
            Limit the colormap or axes to this upper-bound value.

        """
        if not type(name) is str:
            raise TypeError(
                'name argument must be type `str`, but was %s' % type(name))
        self._name = name

        self.cmap = kwargs.pop('cmap', cm.get_cmap('viridis', 64))
        self.label = kwargs.pop('label', None)
        self.norm = kwargs.pop('norm', None)
        self.vmin = kwargs.pop('vmin', None)
        self.vmax = kwargs.pop('vmax', None)

    @property
    def name(self):
        """Name for variable to access from.
        """
        return self._name

    @property
    def cmap(self):
        """Colormap to use to diplay the variable.
        """
        return self._cmap

    @cmap.setter
    def cmap(self, var):
        if type(var) is tuple:
            var, N = var
        else:
            N = 64
        if type(var) is str:
            self._cmap = cm.get_cmap(var, N)
        elif issubclass(type(var), colors.Colormap):
            self._cmap = var
        else:
            raise TypeError

    @property
    def label(self):
        """How to display the variable name when plotting.
        """
        return self._label

    @label.setter
    def label(self, var):
        if type(var) is str:
            self._label = var
        elif not var:
            self._label = self.name
        else:
            raise TypeError

    @property
    def norm(self):
        """???
        """
        return self._norm

    @norm.setter
    def norm(self, var):
        self._norm = var

    @property
    def vmin(self):
        """Limit the colormap or axes to this lower-bound value.
        """
        return self._vmin

    @vmin.setter
    def vmin(self, var):
        self._vmin = var

    @property
    def vmax(self):
        """Limit the colormap or axes to this upper-bound value.
        """
        return self._vmax

    @vmax.setter
    def vmax(self, var):
        self._vmax = var


class VariableSet(object):
    """A default set of properties for variables.

    This makes it easy to have consistent plots.

    This class defines a dictionary of VariableInfo objects, which store
    default information about that variable, for easy and consistent
    plotting.

    Additionally, you can create a VariableSet, and assign it to multiple
    :obj:`~deltametrics.cube.Cube` instances, so that each shares the same
    colormaps, etc.

    Examples
    --------

    Create a default variable set object.

    .. doctest::

        >>> dm.plot.VariableSet()
        <deltametrics.plot.VariableSet object at 0x...>

    """
    _after_init = False  # for "freezing" instance after init

    def __init__(self, override_dict=None):
        """Initialize the VariableSet.

        Initialize the set with default colormaps.

        .. note::
            It is expected that any attribute added to the `known_list` has a
            valid property defined in this class.

        .. note::
            We need an improved way to document each of the defaults. Want to
            display the colormaps, and the limits, the labels, etc. We want to
            be able to automatically label the image with low, high, labels,
            if there is aboundary norm, what happend to above and below
            values, etc. It's too verbose to do this in every file though,
            need a way to automate it. Either a custom Sphinx directive or mpl
            plot directives with function calls. There seems to be a bug that
            doesn't allow function calls to work for docstrings though, only
            primary documents. Another way could be to just loop all the
            ``known_list`` into a single fig at the end?

        Parameters
        ----------
        override_dict : :obj:`dict`, optional
            Dictionary defining variable-property sets. Dictionary key should
            set the  must be either a string (and then match defined colormaps
            in matplotlib or DeltaMetrics), a new matplotlib Colormap object,
            or an Mx3 numpy array that can be coerced into a linear colormap.
        """

        _added_list = ['net_to_gross']
        self.known_list = io.known_variables() + io.known_coords() + _added_list

        for var in self.known_list:
            # set to defaults defined below (or None if no default)
            setattr(self, var, None)

        if override_dict:  # loop override to set if given
            if not type(override_dict) is dict:
                raise TypeError('Invalid type for "override_dict".'
                                'Must be type dict, but was type: %s '
                                % type(override_dict))
            for var in override_dict:
                setattr(self, var, override_dict[var])

        self._after_init = True

    @property
    def variables(self):
        return self._variables

    def __getitem__(self, var):
        """Get the attribute.

        Variable styling (i.e., the `VariableInfo` instances can be accessed by
        slicing the VariableSet with a string matching the VariableInfo `name`
        field. This enables accessing variables in evaluation, rather than
        explicit typing of variable names.
        """

        return self.__getattribute__(var)

    def __setattr__(self, key, var):
        """Set, with check for types.

        This prevents setting non-VariableInfo attributes.
        """
        if not self._after_init:
            object.__setattr__(self, key, var)
        else:
            if type(var) is VariableInfo or var is None:
                object.__setattr__(self, key, var)
            else:
                raise TypeError(
                    'Can only set attributes of type VariableInfo.')

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, var):
        self._x = VariableInfo('x')

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, var):
        self._y = VariableInfo('y')

    @property
    def time(self):
        """Temporal history style.
        """
        return self._time

    @time.setter
    def time(self, var):
        if not var:
            self._time = VariableInfo('time')
        elif type(var) is VariableInfo:
            self._time = var
        else:
            raise TypeError

    @property
    def eta(self):
        """Bed elevation style.
        """
        return self._eta

    @eta.setter
    def eta(self, var):
        if not var:
            cmap = cm.get_cmap('cividis', 64)
            self._eta = VariableInfo('eta', cmap=cmap,
                                     label='bed elevation')
        elif type(var) is VariableInfo:
            self._eta = var
        else:
            raise TypeError

    @property
    def stage(self):
        """Flow stage style.
        """
        return self._stage

    @stage.setter
    def stage(self, var):
        if not var:
            self._stage = VariableInfo('stage')
        elif type(var) is VariableInfo:
            self._stage = var
        else:
            raise TypeError

    @property
    def depth(self):
        """Flow depth style.
        """
        return self._depth

    @depth.setter
    def depth(self, var):
        if not var:
            cmap = cm.get_cmap('Blues', 64)
            self._depth = VariableInfo('depth', cmap=cmap,
                                       vmin=0, label='flow depth')
        elif type(var) is VariableInfo:
            self._depth = var
        else:
            raise TypeError

    @property
    def discharge(self):
        """Flow discharge style.
        """
        return self._discharge

    @discharge.setter
    def discharge(self, var):
        if not var:
            cmap = cm.get_cmap('winter', 64)
            self._discharge = VariableInfo('discharge', cmap=cmap,
                                           label='flow discharge')
        elif type(var) is VariableInfo:
            self._discharge = var
        else:
            raise TypeError

    @property
    def velocity(self):
        """Flow velocity style.
        """
        return self._velocity

    @velocity.setter
    def velocity(self, var):
        if not var:
            cmap = cm.get_cmap('plasma', 64)
            self._velocity = VariableInfo('velocity', cmap=cmap,
                                          label='flow velocity')
        elif type(var) is VariableInfo:
            self._velocity = var
        else:
            raise TypeError

    @property
    def strata_sand_frac(self):
        """Sand fraction style.
        """
        return self._strata_sand_frac

    @strata_sand_frac.setter
    def strata_sand_frac(self, var):
        if not var:
            sandfrac = colors.ListedColormap(
                ['saddlebrown', 'sienna', 'goldenrod', 'gold'])
            sandfrac.set_under('saddlebrown')
            bn = colors.BoundaryNorm([1e-6, 1], sandfrac.N)
            self._strata_sand_frac = VariableInfo('strata_sand_frac',
                                                  cmap=sandfrac,
                                                  norm=None, vmin=0,
                                                  label='sand fraction')
        elif type(var) is VariableInfo:
            self._strata_sand_frac = var
        else:
            raise TypeError

    @property
    def strata_depth(self):
        return self._strata_depth

    @strata_depth.setter
    def strata_depth(self, var):
        if not var:
            self._strata_depth = VariableInfo('strata_depth')
        elif type(var) is VariableInfo:
            self._strata_depth = var
        else:
            raise TypeError

    @property
    def net_to_gross(self):
        """Net-to-gross style.
        """
        return self._net_to_gross

    @net_to_gross.setter
    def net_to_gross(self, var):
        if not var:
            oranges = cm.get_cmap('Oranges', 64)
            greys = cm.get_cmap('Greys_r', 64)
            whiteblack = cm.get_cmap('Greys', 2)
            combined = np.vstack((greys(np.linspace(0.3, 0.6, 2)),
                                  oranges(np.linspace(0.2, 0.8, 6))))
            ntgcmap = colors.ListedColormap(combined, name='net_to_gross')
            ntgcmap.set_bad("white")
            self._net_to_gross = VariableInfo('net_to_gross',
                                              cmap=ntgcmap,
                                              label='net-to-gross')
        elif type(var) is VariableInfo:
            self.__net_to_gross = var
        else:
            raise TypeError


def cartographic_colormap():
    """Colormap for an elevation map style.

    .. warning::
        Not implemented.

    .. note::
        This should implement `something that looks like this
        <https://matplotlib.org/3.2.1/tutorials/colors/colormapnorms.html#twoslopenorm-different-mapping-on-either-side-of-a-center>`_,
        and should be configured to always setting the break to whatever
        sea-level is (or zero?).
    """

    raise NotImplementedError


def aerial_colormap():
    """Colormap for a pesudorealistic looking aerial shot.

    .. warning::
        Not implemented.
    """

    raise NotImplementedError


def append_colorbar(ci, ax):
    """Append a colorbar, consistently placed.

    Adjusts some parameters of the parent axes as well.

    Parameters
    ----------
    ci : `matplotlib.pyplot.pcolormesh`, `matplotlib.pyplot.ImageAxes`
        The colored object generated via matplotlib that the colormap should
        be stolen from.

    ax : `matplotlib.Axes`
        The instance of axes to place the colorbar next to.

    adjust : :obj:`bool`
        Whether to adjust some minor attributes of the parent axis, for
        presentation.

    Returns
    -------
    cb : `matplotlib.colorbar` instance.
        The colorbar instance created.
    """
    divider = axtk.axes_divider.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cb = plt.colorbar(ci, cax=cax)
    cb.ax.tick_params(labelsize=7)
    ax.use_sticky_edges = False

    return cb


def get_display_arrays(VarInst, data=None):
    """Get arrays for display of Variables.

    Function takes as argument a `VariableInstance` from a `Section` or
    `Planform` and an optional :obj:`data` argument, which specifies the data
    type to return, and gives back arrays of 1) data, 2) display x-coordinates
    and 3) display y-coordinates.

    Parameters
    ----------
    VarInst : :obj:`~deltametrics.section.BaseSectionVariable` subclass
        The `Variable` instance to visualize. May be any subclass of
        :obj:`~deltametrics.section.BaseSectionVariable` or
        :obj:`~deltametrics.plan.BasePlanformVariable`.

    data : :obj:`str`, optional
        The type of data to visualize. Supported options are `'spacetime'`,
        `'preserved'`, and `'stratigraphy'`. Default is to display full
        spacetime plot for variable generated from a `DataCube`, and
        stratigraphy for a `StratigraphyCube` variable. Note that variables
        sourced from a `StratigraphyCube` do not support the `spacetime` or
        `preserved` options, and a variable from `DataCube` will only support
        `stratigraphy` if the Cube has computed "quick" stratigraphy.

    Returns
    -------
    data, X, Y : :obj:`ndarray`
        Three matching-size `ndarray` representing the 1) data, 2) display
        x-coordinates and 3) display y-coordinates.
    """
    import pdb; pdb.set_trace()
    # # #  SectionVariables  # # #
    if issubclass(type(VarInst), section.BaseSectionVariable):
        # #  DataSection  # #
        if isinstance(VarInst, section.DataSectionVariable):
            data = data or VarInst._default_data
            if data in VarInst._spacetime_names:
                return VarInst, VarInst._S, VarInst._Z
            elif data in VarInst._preserved_names:
                return VarInst.as_preserved(), VarInst._S, VarInst._Z
            elif data in VarInst._stratigraphy_names:
                _sp = VarInst.as_stratigraphy()
                _den = _sp.toarray().view(section.DataSectionVariable)
                _arr_Y = VarInst.strat_attr['psvd_flld'][:_sp.shape[0], ...]
                _arr_X = np.tile(VarInst._s, (_sp.shape[0], 1))
                return _den[1:, 1:], _arr_X, _arr_Y
            else:
                raise ValueError('Bad data argument: %s' % str(data))
        # #  StratigraphySection  # #
        elif isinstance(VarInst, section.StratigraphySectionVariable):
            data = data or VarInst._default_data
            if data in VarInst._spacetime_names:
                VarInst._check_knows_spacetime()  # always False
            elif data in VarInst._preserved_names:
                VarInst._check_knows_spacetime()  # always False
            elif data in VarInst._stratigraphy_names:
                return VarInst, VarInst._S, VarInst._Z
            else:
                raise ValueError('Bad data argument: %s' % str(data))
        else:
            raise TypeError

    # # #  PlanformVariables  # # #
    elif False:  # issubclass(type(VarInst), plan.BasePlanformVariable):
        raise NotImplementedError
    else:
        raise TypeError('Invaid "VarInst" type: %s' % type(VarInst))


def get_display_lines(VarInst, data=None):
    """Get lines for display of Variables.

    Function takes as argument a `VariableInstance` from a `Section` or
    `Planform` and an optional :obj:`data` argument, which specifies the data
    type to return, and gives back data and line segments for display.

    Parameters
    ----------
    VarInst : :obj:`~deltametrics.section.BaseSectionVariable` subclass
        The `Variable` instance to visualize. May be any subclass of
        :obj:`~deltametrics.section.BaseSectionVariable` or
        :obj:`~deltametrics.plan.BasePlanformVariable`.

    data : :obj:`str`, optional
        The type of data to visualize. Supported options are `'spacetime'`,
        `'preserved'`, and `'stratigraphy'`. Default is to display full
        spacetime plot for variable generated from a `DataCube`.  and
        stratigraphy for a `StratigraphyCube` variable. Variables from
        `DataCube` will only support `stratigraphy` if the Cube has computed
        "quick" stratigraphy.

    .. note::
        Not currently implemented for variables from the `StratigraphyCube`.

    Returns
    -------
    vals, segments : :obj:`ndarray`
        An off-by-one sized array of data values (`vals`) to color the line
        segments (`segments`). The segments are organized along the 0th
        dimension of the array.
    """
    # # #  SectionVariables  # # #
    if issubclass(type(VarInst), section.BaseSectionVariable):
        # #  DataSection  # #
        if isinstance(VarInst, section.DataSectionVariable):
            def _reshape_long(X):
                # util for reshaping s- and z-values appropriately
                return np.vstack((X[:, :-1].flatten(),
                                  X[:, 1:].flatten())).T.reshape(-1, 2, 1)
            data = data or VarInst._default_data
            if data in VarInst._spacetime_names:
                z = _reshape_long(VarInst._Z)
                vals = VarInst[:, :-1]
            elif data in VarInst._preserved_names:
                z = _reshape_long(VarInst._Z)
                vals = VarInst.as_preserved()[:, :-1]
            elif data in VarInst._stratigraphy_names:
                VarInst._check_knows_stratigraphy()  # need to check explicitly
                z = _reshape_long(np.copy(VarInst.strat_attr['strata']))
                vals = VarInst[:, :-1]
            else:
                raise ValueError('Bad data argument: %s' % str(data))
            s = _reshape_long(VarInst._S)
            segments = np.concatenate([s, z], axis=2)
            if data in VarInst._stratigraphy_names:
                # flip = draw late to early
                vals = np.fliplr(np.flipud(vals))
                segments = np.flipud(segments)
            return vals, segments
        # #  StratigraphySection  # #
        elif isinstance(VarInst, section.StratigraphySectionVariable):
            data = data or VarInst._default_data
            if data in VarInst._spacetime_names:
                VarInst._check_knows_spacetime()  # always False
            elif data in VarInst._preserved_names:
                VarInst._check_knows_spacetime()  # always False
            elif data in VarInst._stratigraphy_names:
                raise NotImplementedError  # not sure best implementation
            else:
                raise ValueError('Bad data argument: %s' % str(data))
        else:
            raise TypeError

    # # #  PlanformVariables  # # #
    elif False:  # issubclass(type(VarInst), plan.BasePlanformVariable):
        raise NotImplementedError
    else:
        raise TypeError('Invaid "VarInst" type: %s' % type(VarInst))


def get_display_limits(VarInst, data=None):
    """Get limits to resize the display of Variables.

    Function takes as argument a `VariableInstance` from a `Section` or
    `Planform` and an optional :obj:`data` argument, which specifies how to
    determine the limits to return.

    Parameters
    ----------
    VarInst : :obj:`~deltametrics.section.BaseSectionVariable` subclass
        The `Variable` instance to visualize. May be any subclass of
        :obj:`~deltametrics.section.BaseSectionVariable` or
        :obj:`~deltametrics.plan.BasePlanformVariable`.

    data : :obj:`str`, optional
        The type of data to compute limits for. Typically this will be the
        same value used with either :obj:`get_display_arrays` or
        :obj:`get_display_lines`. Supported options are `'spacetime'`,
        `'preserved'`, and `'stratigraphy'`.

    Returns
    -------
    xmin, xmax, ymin, ymax : :obj:`float`
        Values to use as limits on a plot. Use with, for example,
        ``ax.set_xlim((xmin, xmax))``.
    """
    # # #  SectionVariables  # # #
    if issubclass(type(VarInst), section.BaseSectionVariable):
        # #  DataSection  # #
        if isinstance(VarInst, section.DataSectionVariable):
            data = data or VarInst._default_data
            if data in VarInst._spacetime_names:
                return np.min(VarInst._S), np.max(VarInst._S), \
                    np.min(VarInst._Z), np.max(VarInst._Z)
            elif data in VarInst._preserved_names:
                VarInst._check_knows_stratigraphy()  # need to check explicitly
                return np.min(VarInst._S), np.max(VarInst._S), \
                    np.min(VarInst._Z), np.max(VarInst._Z)
            elif data in VarInst._stratigraphy_names:
                VarInst._check_knows_stratigraphy()  # need to check explicitly
                _strata = np.copy(VarInst.strat_attr['strata'])
                return np.min(VarInst._S), np.max(VarInst._S), \
                    np.min(_strata), np.max(_strata) * 1.5
            else:
                raise ValueError('Bad data argument: %s' % str(data))

        # #  StratigraphySection  # #
        elif isinstance(VarInst, section.StratigraphySectionVariable):
            data = data or VarInst._default_data
            if data in VarInst._spacetime_names:
                VarInst._check_knows_spacetime()  # always False
            elif data in VarInst._preserved_names:
                VarInst._check_knows_spacetime()  # always False
            elif data in VarInst._stratigraphy_names:
                return np.min(VarInst._S), np.max(VarInst._S), \
                    np.min(VarInst._Z), np.max(VarInst._Z) * 1.5
            else:
                raise ValueError('Bad data argument: %s' % str(data))

        else:
            raise TypeError

    # # #  PlanformVariables  # # #
    elif False:  # issubclass(type(VarInst), plan.BasePlanformVariable):
        raise NotImplementedError
    else:
        raise TypeError('Invaid "VarInst" type: %s' % type(VarInst))


def _fill_steps(where, x=1, y=1, y0=0, **kwargs):
    """Fill rectangles where the boolean indicates ``True``.

    Creates an :obj:`x` by :obj:`y` :obj:`matplotlib.patches.Rectangle` at
    each index where the index `i` in boolean :obj:`where` indicates, with the
    lower left of the patch at (`i`, :obj:`y0`).

    This utility function is utilized internally. Most often, it is used to
    highlight where time has been preserved (or eliminated) in a timeseries of
    data. For example, see
    :obj:`~deltametrics.plot.show_one_dimensional_trajectory_to_strata`.

    Parameters
    ----------
    where : :obj:`ndarray`
        Boolean `numpy` `ndarray` indicating which locations to create a patch
        at.

    x : :obj:`float`
        The x-direction width of the `Rectangles`.

    y : :obj:`float`
        The y-direction height of the `Rectangles`.

    y0 : :obj:`float`
        The y-direction origin (i.e., lower-left y-value) of the `Rectangles`.

    **kwargs
        Additional `matplotlib` keyword arguments passed to the
        `Rectangle` instantiation. This is where you can set most
        `matplotlib` `**kwargs` (e.g., `color`, `edgecolor`).

    Returns
    -------
    pc : :obj:`matplotlib.patches.PatchCollection`
        Collection of `Rectangle` `Patch` objects.
    """
    pl = []
    for i, pp in enumerate(np.argwhere(where[1:]).flatten()):
        _r = ptch.Rectangle((pp, y0), x, y, **kwargs)
        pl.append(_r)
    return coll.PatchCollection(pl, match_original=True)


def show_one_dimensional_trajectory_to_strata(e, dz=0.05, z=None, ax=None):
    """1d elevation to stratigraphy.

    This function creates and displays a one-dimensional elevation timeseries
    as a trajectory of elevations, resultant stratigraphy, and time preserved
    in "boxy" stratigraphy. The function is helpful for description of
    algorithms to compute stratigraphy, and for debugging and checking outputs
    from computations.

    For example, we can quickly visualize the processing of a 1D timeseries of
    bed elevations into boxy stratigraphy with this routine.

    .. plot:: guides/userguide_1d_example.py
        :include-source:

    The orange line depicts the resultant stratigraphy, with all
    bed-elevations above this line cut from the stratigraphy by the
    stratigraphic filter.  The column on the right records which
    time-interval is recorded in the stratigraphy at each elevation.

    Parameters
    ----------
    e : :obj:`ndarray`
        Elevation data as a 1D array.

    dz : :obj:`float`, optional
        Vertical grid resolution.

    z : :obj:`ndarray`, optional
        Vertical grid. Must specify ``dz=None`` to use this option.

    ax : :obj:`matplotlib.pyplot.axes`, optional
        Axes to plot into. A figure and axes is created, if not given.

    Returns
    -------
    """
    # reprocess shape to be 1d array, if needed.
    if e.ndim > 1:
        if e.shape[1:] == (1, 1):
            e = e.squeeze()
        else:
            raise ValueError('Elevation data "e" must be one-dimensional.')
    t = np.arange(e.shape[0])  # x-axis time array
    t3 = np.expand_dims(t, axis=(1, 2))  # 3d time, for slicing

    z = strat._determine_strat_coordinates(e, dz=dz, z=z)  # vert coordinates
    s, p = strat._compute_elevation_to_preservation(e)  # strat, preservation
    sc, dc = strat._compute_preservation_to_cube(s, z)
    lst = np.argmin(s < s[-1])  # last elevation

    c = np.full_like(z, np.nan)
    c[sc[:, 0]] = t[dc[:, 0]]
    cp = np.tile(c, (2, 1)).T

    # make the plots
    if not ax:
        fig, ax = plt.subplots()
    # timeseries plot
    pt = np.zeros_like(t)  # for psvd timesteps background
    pt[np.union1d(p.nonzero()[0], np.array(
        strat._compute_preservation_to_time_intervals(p).nonzero()[0]))] = 1
    ax.add_collection(_fill_steps(p, x=1, y=np.max(e) - np.min(e),
                                  y0=np.min(e), facecolor='0.8'))
    _ppc = ax.add_patch(ptch.Rectangle((0, 0), 0, 0, facecolor='0.8',
                                       label='psvd timesteps'))  # add for lgnd
    _ss = ax.hlines(e[p], 0, e.shape[0], linestyles='dashed', colors='0.7')
    _l = ax.axvline(lst, c='k')
    _e = ax.step(t, e, where='post', label='elevation')
    _s = ax.step(t, s, linestyle='--', where='post', label='stratigraphy')
    _pd = ax.plot(t[p], s[p], color='0.5', marker='o',
                  ls='none', label='psvd time')
    ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    ax.grid(which='both', axis='x')

    # boxy strata plot
    divider = axtk.make_axes_locatable(ax)
    ax_s = divider.append_axes("right", 0.5, pad=0.1, sharey=ax)
    ax_s.yaxis.tick_right()
    ax_s.xaxis.set_visible(False)
    __x, __y = np.meshgrid(np.array([0, 1]), z)
    _colmap = plt.cm.get_cmap('viridis', e.shape[0])
    _c = ax_s.pcolormesh(__x, __y, cp,
                         cmap=_colmap, vmin=0, vmax=e.shape[0], shading='auto')
    _ss2 = ax_s.hlines(e[p], 0, 1, linestyles='dashed', colors='gray')
    _cstr = [str(int(cc)) if np.isfinite(cc) else 'nan' for cc in c.flatten()]
    for i, __cstr in enumerate(_cstr):
        ax_s.text(0.3, z[i], str(__cstr), fontsize=8)

    # adjust and add legend
    if np.any(e < 0):
        yView = np.absolute(e).max() * 1.2
        ax.set_ylim(np.min(e) * 1.2, np.maximum(0, np.max(e) * 1.2))
    else:
        ax.set_ylim(np.min(e) * 0.8, np.max(e) * 1.2)
    ax.legend()
