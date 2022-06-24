import numpy as np
import xarray as xr

import colorsys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as ptch
import matplotlib.collections as coll
import mpl_toolkits.axes_grid1 as axtk

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
        self.known_list = ['eta', 'stage', 'depth', 'discharge',
                           'velocity', 'sedflux', 'strata_sand_frac',
                           'sandfrac'] + \
                          ['x', 'y', 'time'] + _added_list

        self._variables = []
        for var in self.known_list:
            # set to defaults defined below (or None if no default)
            setattr(self, var, None)
            self._variables.append(var)

        if override_dict:  # loop override to set if given
            if not type(override_dict) is dict:
                raise TypeError('Invalid type for "override_dict".'
                                'Must be type dict, but was type: %s '
                                % type(override_dict))
            for var in override_dict:
                setattr(self, var, override_dict[var])
                self._variables.append(var)

        self._after_init = True

    @property
    def variables(self):
        """Variables known to the VariableSet."""
        return self._variables

    def __getitem__(self, var):
        """Get the attribute.

        Variable styling (i.e., the `VariableInfo` instances can be accessed by
        slicing the VariableSet with a string matching the VariableInfo `name`
        field. This enables accessing variables in evaluation, rather than
        explicit typing of variable names.

        Parameters
        ----------
        variable : :obj:`str`
            Which variable to get the `VariableInfo` for.

        .. note::

            If `variable` is not identified in the VariableSet, then a default
            instance of :obj:`VariableInfo` is instantiated with the variable
            name and returned.
        """
        if var in self._variables:
            return self.__getattribute__(var)
        else:
            return VariableInfo(var)

    def __setattr__(self, key, var):
        """Set, with check for types.

        This prevents setting non-VariableInfo attributes.
        """
        if not self._after_init:
            object.__setattr__(self, key, var)
        else:
            if type(var) is VariableInfo or var is None:
                object.__setattr__(self, key, var)
                # add it to the list of variables
                self._variables.append(key)
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
    def sedflux(self):
        """Flow sedflux style.
        """
        return self._sedflux

    @sedflux.setter
    def sedflux(self, var):
        if not var:
            cmap = cm.get_cmap('magma', 64)
            self._sedflux = VariableInfo('sedflux', cmap=cmap,
                                         label='sediment flux')
        elif type(var) is VariableInfo:
            self._sedflux = var
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
            self._strata_sand_frac = VariableInfo('strata_sand_frac',
                                                  cmap=sandfrac,
                                                  norm=None, vmin=0,
                                                  label='sand fraction')
        elif type(var) is VariableInfo:
            self._strata_sand_frac = var
        else:
            raise TypeError

    @property
    def sandfrac(self):
        """Sand fraction style.
        """
        return self._sandfrac

    @sandfrac.setter
    def sandfrac(self, var):
        if not var:
            ends_str = ['saddlebrown',  'gold']  # define end points as strs
            endpts_colors = [matplotlib.colors.to_rgb(col) for col in ends_str]
            endpts_colors = np.column_stack(  # interpolate 64 between end pts
                [np.linspace(endpts_colors[0][i], endpts_colors[1][i], num=64)
                 for i in range(3)]  # for each column in RGB
                 )
            sand_frac = matplotlib.colors.ListedColormap(endpts_colors)
            sand_frac.set_under('saddlebrown')
            self._sandfrac = VariableInfo('sandfrac',
                                          cmap=sand_frac,
                                          norm=None, vmin=0,
                                          label='sand fraction')
        elif type(var) is VariableInfo:
            self._sandfrac = var
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
            ntgcmap.set_bad("white", alpha=0)
            self._net_to_gross = VariableInfo('net_to_gross',
                                              cmap=ntgcmap,
                                              label='net-to-gross')
        elif type(var) is VariableInfo:
            self.__net_to_gross = var
        else:
            raise TypeError


def cartographic_colormap(H_SL=0.0, h=4.5, n=1.0):
    """Colormap for an elevation map style.

    Parameters
    ----------
    H_SL : :obj:`float`, optional
        Sea level for the colormap. This is the break-point
        between blues and greens. Default value is `0.0`.

    h : :obj:`float`, optional
        Channel depth for the colormap. This is some characteristic *below
        sea-level* relief for the colormap to extend to through the range of
        blues. Default value is `4.5`.

    n : :obj:`float`, optional
        Surface topography relief for the colormap. This is some
        characteristic *above sea-level* relief for the colormap to extend to
        through the range of greens. Default value is `1.0`.

    Returns
    -------
    delta : :obj:`matplotib.colors.ListedColormap`
        The colormap object, which can then be used by other `matplotlib`
        plotting routines (see examples below).

    norm : :obj:`matplotib.colors.BoundaryNorm`
        The color normalization object, which can then be used by other
        `matplotlib` plotting routines (see examples below).

    Examples
    --------

    To display with default depth and relief parameters (left) and with adjust
    parameters to highlight depth variability (right):

    .. plot::
        :include-source:

        golfcube = dm.sample_data.golf()

        cmap0, norm0 = dm.plot.cartographic_colormap(H_SL=0)
        cmap1, norm1 = dm.plot.cartographic_colormap(H_SL=0, h=5, n=0.5)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        im0 = ax[0].imshow(golfcube['eta'][-1, ...], origin='lower',
                       cmap=cmap0, norm=norm0)
        cb0 = dm.plot.append_colorbar(im0, ax[0])
        im1 = ax[1].imshow(golfcube['eta'][-1, ...], origin='lower',
                       cmap=cmap1, norm=norm1)
        cb1 = dm.plot.append_colorbar(im1, ax[1])
        plt.show()
    """
    blues = matplotlib.cm.get_cmap('Blues_r', 64)
    greens = matplotlib.cm.get_cmap('YlGn_r', 64)
    combined = np.vstack((blues(np.linspace(0.1, 0.7, 5)),
                          greens(np.linspace(0.2, 0.8, 5))))
    delta = matplotlib.colors.ListedColormap(combined, name='delta')
    bounds = np.hstack(
        (np.linspace(H_SL-h, H_SL-(n/2), 5),
         np.linspace(H_SL, H_SL+n, 6)))
    norm = matplotlib.colors.BoundaryNorm(bounds, len(bounds)-1)
    return delta, norm


def aerial_colormap():
    """Colormap for a pesudorealistic looking aerial shot.

    .. warning::
        Not implemented.
    """

    raise NotImplementedError


def append_colorbar(ci, ax, size=2, pad=2, labelsize=9, **kwargs):
    """Append a colorbar, consistently placed.

    Adjusts some parameters of the parent axes as well.

    Parameters
    ----------
    ci : `matplotlib.pyplot.pcolormesh`, `matplotlib.pyplot.ImageAxes`
        The colored object generated via matplotlib that the colormap should
        be stolen from.

    ax : `matplotlib.Axes`
        The instance of axes to place the colorbar next to.

    size : :obj:`float`, optional
        Width (percent of parent axis width) of the colorbar; default is 2.

    pad : :obj:`float`, optional
        Padding between parent and colorbar axes. Default is 0.05.

    labelsize : :obj:`int`, optional
        Font size of label text. Default is 9pt.

    **kwargs
        Passed to `matplotlib.pyplot.colorbar`.

    Returns
    -------
    cb : `matplotlib.colorbar` instance.
        The colorbar instance created.
    """
    divider = axtk.axes_divider.make_axes_locatable(ax)
    cax = divider.append_axes("right", size=str(size)+"%", pad=str(pad)+"%")
    cb = plt.colorbar(ci, cax=cax, **kwargs)
    cb.ax.tick_params(labelsize=labelsize)
    ax.use_sticky_edges = False

    return cb

def style_axes_km(*args):
    """Style axes with kilometers, when initially set as meters.

    This function can be used two ways. Passing an `Axes` object as the first
    argument and optionally a string specifying `'x'`, `'y'`, or `'xy'`
    [default] will chnage the appearance of the `Axes` object. Alternatively,
    the function can be specified as the `ticker` function when setting a
    label formatter.

    Parameters
    ----------
    ax : :obj:`matplotlib.axes.Axes`
        Axes object to format

    which : :obj:`str`, optional
        Which axes to style as kilometers. Default is `xy`, but optionally
        pass `x` or `y` to only style one axis.

    x : :obj:`float`
        If using as a function to format labels, the tick mark location.

    Examples
    --------

    .. plot::
        :include-source:
        :context: reset

        golf = dm.sample_data.golf()

        fig, ax = plt.subplots(
            3, 1,
            gridspec_kw=dict(hspace=0.5))
        golf.quick_show('eta', ax=ax[0], ticks=True)

        golf.quick_show('eta', ax=ax[1], ticks=True)
        dm.plot.style_axes_km(ax[1])

        golf.quick_show('eta', axis=1, idx=10, ax=ax[2])
        ax[2].xaxis.set_major_formatter(dm.plot.style_axes_km)
        # OR use: dm.plot.style_axes_km(ax[2], 'x')

        plt.show()
    """
    if isinstance(args[0], matplotlib.axes.Axes):
        ax = args[0]
        if len(args) > 1:
            which = args[1]
        else:
            # default is to apply to both xy
            which = 'xy'
        # recursive calls to this func!
        if 'x' in which:
            ax.xaxis.set_major_formatter(style_axes_km)
        if 'y' in which:
            ax.yaxis.set_major_formatter(style_axes_km)

    else:
        v = args[0]
        return f'{v / 1000.:g}'


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
    # # #  SectionVariables  # # #
    if VarInst.slicetype == 'data_section':
        # #  DataSection  # #
        data = data or 'spacetime'
        if data in VarInst.strat._spacetime_names:
            _S, _Z = np.meshgrid(VarInst['s'], VarInst[VarInst.dims[0]])
            return VarInst.values, _S, _Z
        elif data in VarInst.strat._preserved_names:
            VarInst.strat._check_knows_spacetime()
            _S, _Z = np.meshgrid(VarInst['s'], VarInst[VarInst.dims[0]])
            return VarInst.strat.as_preserved(), _S, _Z
        elif data in VarInst.strat._stratigraphy_names:
            _sp = VarInst.strat.as_stratigraphy()
            _den = _sp.toarray()  # .view(section.DataSectionVariable)
            _arr_Y = VarInst.strat.strat_attr['psvd_flld'][:_sp.shape[0], ...]
            _arr_X = np.tile(VarInst['s'], (_sp.shape[0], 1))
            return _den[1:, 1:], _arr_X, _arr_Y
        else:
            raise ValueError('Bad data argument: %s' % str(data))

    elif VarInst.slicetype == 'stratigraphy_section':
        # #  StratigraphySection  # #
        data = data or 'stratigraphy'
        if data in VarInst.strat._spacetime_names:
            VarInst.strat._check_knows_spacetime()  # always False
        elif data in VarInst.strat._preserved_names:
            VarInst.strat._check_knows_spacetime()  # always False
        elif data in VarInst.strat._stratigraphy_names:
            _S, _Z = np.meshgrid(VarInst['s'], VarInst[VarInst.dims[0]])
            return VarInst, _S, _Z
        else:
            raise ValueError('Bad data argument: %s' % str(data))

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
    if VarInst.slicetype == 'data_section':
        # #  DataSection  # #
        def _reshape_long(X):
            # util for reshaping s- and z-values appropriately
            return np.vstack((X[:, :-1].flatten(),
                              X[:, 1:].flatten())).T.reshape(-1, 2, 1)
        data = data or 'spacetime'
        _S, _Z = np.meshgrid(VarInst['s'], VarInst[VarInst.dims[0]])
        if data in VarInst.strat._spacetime_names:
            z = _reshape_long(_Z)
            vals = VarInst[:, :-1]
        elif data in VarInst.strat._preserved_names:
            z = _reshape_long(_Z)
            vals = VarInst.strat.as_preserved()[:, :-1]
        elif data in VarInst.strat._stratigraphy_names:
            VarInst.strat._check_knows_stratigraphy()  # need to check explicitly
            z = _reshape_long(np.copy(VarInst.strat.strat_attr['strata']))
            vals = VarInst[:, :-1]
        else:
            raise ValueError('Bad data argument: %s' % str(data))
        s = _reshape_long(_S)
        segments = np.concatenate([s, z], axis=2)
        if data in VarInst.strat._stratigraphy_names:
            # flip = draw late to early
            vals = np.fliplr(np.flipud(vals))
            segments = np.flipud(segments)
        return np.array(vals), segments

    elif VarInst.slicetype == 'stratigraphy_section':
        # #  StratigraphySection  # #
        data = data or 'stratigraphy'
        if data in VarInst.strat._spacetime_names:
            VarInst.strat._check_knows_spacetime()  # always False
        elif data in VarInst.strat._preserved_names:
            VarInst.strat._check_knows_spacetime()  # always False
        elif data in VarInst.strat._stratigraphy_names:
            raise NotImplementedError  # not sure best implementation
        else:
            raise ValueError('Bad data argument: %s' % str(data))

    # # #  PlanformVariables  # # #
    elif False:  # issubclass(type(VarInst), plan.BasePlanformVariable):
        raise NotImplementedError
    else:
        raise TypeError('Invaid "VarInst" type: %s' % type(VarInst))


def get_display_limits(VarInst, data=None, factor=1.5):
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

    factor : :obj:`float`, optional
        Factor to extend vertical limits upward for stratigraphic sections.

    Returns
    -------
    xmin, xmax, ymin, ymax : :obj:`float`
        Values to use as limits on a plot. Use with, for example,
        ``ax.set_xlim((xmin, xmax))``.
    """
    # # #  SectionVariables  # # #
    if VarInst.slicetype == 'data_section':
        # #  DataSection  # #
        data = data or 'spacetime'
        _S, _Z = np.meshgrid(VarInst['s'], VarInst[VarInst.dims[0]])
        if data in VarInst.strat._spacetime_names:
            return np.min(_S), np.max(_S), \
                np.min(_Z), np.max(_Z)
        elif data in VarInst.strat._preserved_names:
            VarInst.strat._check_knows_stratigraphy()  # need to check explicitly
            return np.min(_S), np.max(_S), \
                np.min(_Z), np.max(_Z)
        elif data in VarInst.strat._stratigraphy_names:
            VarInst.strat._check_knows_stratigraphy()  # need to check explicitly
            _strata = np.copy(VarInst.strat.strat_attr['strata'])
            return np.min(_S), np.max(_S), \
                np.min(_strata), np.max(_strata) * factor
        else:
            raise ValueError('Bad data argument: %s' % str(data))

    elif VarInst.slicetype == 'stratigraphy_section':
        # #  StratigraphySection  # #
        data = data or 'stratigraphy'
        _S, _Z = np.meshgrid(VarInst['s'], VarInst[VarInst.dims[0]])
        if data in VarInst.strat._spacetime_names:
            VarInst.strat._check_knows_spacetime()  # always False
        elif data in VarInst.strat._preserved_names:
            VarInst.strat._check_knows_spacetime()  # always False
        elif data in VarInst.strat._stratigraphy_names:
            return np.min(_S), np.max(_S), \
                np.min(_Z), np.max(_Z) * factor
        else:
            raise ValueError('Bad data argument: %s' % str(data))

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


def show_one_dimensional_trajectory_to_strata(e, sigma_dist=None,
                                              dz=None, z=None,
                                              nz=None, ax=None,
                                              show_strata=True,
                                              label_strata=False):
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

    sigma_dist : :obj:`ndarray`, :obj:`float`, :obj:`int`, optional
        Optional subsidence distance argument used to adjust the elevation
        data to account for subsidence when computing stratigraphy. See
        :obj:`_adjust_elevation_by_subsidence` for a complete description.

    z : :obj:`ndarray`, optional
        Vertical coordinates for stratigraphy, in meters. Optional, and
        mutually exclusive with :obj:`dz` and :obj:`nz`,
        see :obj:`_determine_strat_coordinates` for complete description.

    dz : :obj:`float`, optional
        Vertical resolution of stratigraphy, in meters. Optional, and mutually
        exclusive with :obj:`z` and :obj:`nz`,
        see :obj:`_determine_strat_coordinates` for complete description.

    nz : :obj:`int`, optional
        Number of intervals for vertical coordinates of stratigraphy.
        Optional, and mutually exclusive with :obj:`z` and :obj:`dz`,
        see :obj:`_determine_strat_coordinates` for complete description.

    ax : :obj:`matplotlib.pyplot.axes`, optional
        Axes to plot into. A figure and axes is created, if not given.

    show_strata : :obj:`bool`, optional
        Whether to plot the resultant stratigraphy as a strip log on a second
        axis on the right side. This axis included numbers indicating the
        timestep for each voxel of preserved stratigraphy.

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
    e_in = e.copy()

    if sigma_dist is not None:
        # adjust elevations by subsidence rate
        e = strat._adjust_elevation_by_subsidence(e_in, sigma_dist)
    s, p = strat._compute_elevation_to_preservation(e)  # strat, preservation
    z = strat._determine_strat_coordinates(e, dz=dz, z=z, nz=nz)  # vert coordinates
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
    ax.add_collection(_fill_steps(p, x=1, y=np.max(e_in) - np.min(e),
                                  y0=np.min(e), facecolor='0.8'))
    ax.add_patch(ptch.Rectangle((0, 0), 0, 0, facecolor='0.8',
                                label='psvd timesteps'))  # add for lgnd
    ax.hlines(s[p], 0, e.shape[0], linestyles='dashed', colors='0.7')
    ax.axvline(lst, c='k')
    ax.step(t, e_in, where='post', label='elevation')
    ax.step(t, s, linestyle='--', where='post', label='stratigraphy')
    ax.plot(t[p], s[p], color='0.5', marker='o',
            ls='none', label='psvd time')
    if len(t) < 100:
        ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    ax.grid(which='both', axis='x')

    if show_strata:
        # boxy strata plot
        divider = axtk.make_axes_locatable(ax)
        ax_s = divider.append_axes("right", 0.5, pad=0.1, sharey=ax)
        ax_s.yaxis.tick_right()
        ax_s.xaxis.set_visible(False)
        __x, __y = np.meshgrid(np.array([0, 1]), z)
        _colmap = plt.cm.get_cmap('viridis', e.shape[0])
        ax_s.pcolormesh(__x, __y, cp,
                        cmap=_colmap, vmin=0, vmax=e.shape[0],
                        shading='auto')
        ax_s.hlines(e[p], 0, 1, linestyles='dashed', colors='gray')
        _cstr = [str(int(cc)) if np.isfinite(cc) else 'nan' for cc in c.flatten()]
        ax_s.set_xlim(0, 1)
        if label_strata:
            for i, __cstr in enumerate(_cstr):
                ax_s.text(0.3, z[i], str(__cstr), fontsize=8)

    # adjust and add legend
    if np.any(e < 0):
        ax.set_ylim(np.min(e) * 1.2, np.maximum(0, np.max(e_in) * 1.2))
    else:
        ax.set_ylim(np.min(e) * 0.8, np.max(e_in) * 1.2)
    ax.legend()


def _scale_lightness(rgb, scale_l):
    """Utility to scale the lightness of some color.

    Make a color perceptually lighter or darker. Adapted from
    https://stackoverflow.com/a/60562502/4038393.

    Parameters
    ----------
    rgb : :obj:`tuple`
        A three element tuple of the RGB values for the color.

    scale_l : :obj:`float`
        The scale factor, relative to a value of `1`. A value of 1 performs no
        scaling, and values less than 1 darken the color, whereas values
        greater than 1 brighten the color.

    Returns
    -------
    scaled : :obj:`tuple`
        Scaled color RGB tuple.

    Example
    -------

    .. plot::

        fig, ax = plt.subplots(figsize=(5, 2))

        # initial color red
        red = (1.0, 0.0, 0.0)
        ax.plot(-1, 1, 'o', color=red)

        # scale from 1 to 0.05
        scales = np.arange(1, 0, -0.05)

        # loop through scales and plot
        for s, scale in enumerate(scales):
            darker_red = dm.plot._scale_lightness(red, scale)
            ax.plot(s, scale, 'o', color=darker_red)

        plt.show()
    """
    # https://stackoverflow.com/a/60562502/4038393
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * np.abs(scale_l)), s=s)


def show_histograms(*args, sets=None, ax=None, **kwargs):
    """Show multiple histograms, including as sets.

    Parameters
    ----------
    *args : :obj:`tuple`
        Any number of comma separated tuples, where each tuple is a set of
        `(counts, bins)`, for example, as an output from `np.histogram()`.

    sets : :obj:`list`, optional
        A list or numpy array indicating the set each pdf belongs to. For
        example, [0, 0, 1, 1, 2] incidates the first two `*args` are from the
        first set, the third and fourth belong to a second set, and the fifth
        argument belongs to a third set. Length of `sets` must match the
        number of comma separated `*args`. If not supplied, all histograms are
        colored differently (up to 10).

    ax : :obj:`matplotlib.pyplot.axes`, optional
        Axes to plot into. A figure and axes is created, if not given.

    **kwargs
        Additional `matplotlib` keyword arguments passed to the
        `bar` plotting routine. In current implementation, cannot use
        arguments `width`, `edgecolor`, or `facecolor`.

    Returns
    -------

    Examples
    --------

    .. plot::
        :include-source:

        locs = [0.25, 1, 0.5, 4, 2]
        scales = [0.1, 0.25, 0.4, 0.5, 0.1]
        bins = np.linspace(0, 6, num=40)

        hist_bin_sets = [np.histogram(np.random.normal(l, s, size=500), bins=bins, density=True) for l, s in zip(locs, scales)]

        fig, ax = plt.subplots()
        dm.plot.show_histograms(*hist_bin_sets, sets=[0, 1, 0, 1, 2], ax=ax)
        ax.set_xlim((0, 6))
        ax.set_ylabel('density')
        plt.show()
    """
    if not ax:
        fig, ax = plt.subplots()

    if (sets is None):
        n_sets = len(args)
        sets = np.arange(n_sets)
    else:
        n_sets = len(np.unique(sets))
        sets = np.array(sets)

    if len(sets) != len(args):
        raise ValueError(
            'Number of histogram tuples must match length of `sets` list.')

    for i in range(n_sets):
        CN = 'C%d' % (i)
        match = np.where((sets == i))[0]
        scales = np.linspace(0.8, 1.2, num=len(match))
        CNs = [_scale_lightness(colors.to_rgb(CN), sc) for s, sc in enumerate(scales)]
        for n in range(len(match)):
            hist, bins = args[match[n]]
            bin_width = (bins[1:] - bins[:-1])
            bin_cent = bins[:-1] + (bin_width/2)
            ax.bar(bin_cent, hist, width=bin_width,
                   edgecolor=CNs[n], facecolor=CNs[n], **kwargs)


def aerial_view(elevation_data, datum=0, ax=None, ticks=False,
                colorbar_kw={}, return_im=False, **kwargs):
    """Show an aerial plot of an elevation dataset.

    See also: implementation wrapper for a cube.

    Parameters
    ----------

    elevation_data
        2D array of elevations.

    datum
        Sea level reference, default value is 0.

    ax : :obj:`~matplotlib.pyplot.Axes` object, optional
        A `matplotlib` `Axes` object to plot the section. Optional; if not
        provided, a call is made to ``plt.gca()`` to get the current (or
        create a new) `Axes` object.

    ticks
        Whether to show ticks. Default is false, no tick labels.

    colorbar_kw
        Dictionary of keyword args passed to :func:`append_colorbar`.

    return_im : bool, optional
        Returns the ``plt.imshow()`` image object if True. Default is False.

    **kwargs
        Optionally, arguments accepted by :func:`cartographic_colormap`, or
        `imshow`.

    Returns
    -------
    colorbar
        Colorbar instance appended to the axes.

    im : :obj:`~matplotlib.image.AxesImage` object, optional
        Optional return of the image object itself if ``return_im`` is True


    Examples
    --------
    .. plot::
        :include-source:

        golfcube = dm.sample_data.golf()
        elevation_data = golfcube['eta'][-1, :, :]

        fig, ax = plt.subplots()
        dm.plot.aerial_view(elevation_data, ax=ax)
        plt.show()

    """
    if not ax:
        fig, ax = plt.subplots()

    # process to make a cmap
    h = kwargs.pop('h', 3)
    n = kwargs.pop('n', 1)
    carto_cm, carto_norm = cartographic_colormap(H_SL=0, h=h, n=n)

    # get the extent to plot
    if isinstance(elevation_data, xr.core.dataarray.DataArray):
        d0, d1 = elevation_data.dims
        d0_arr, d1_arr = elevation_data[d0], elevation_data[d1]
        _extent = [d1_arr[0],                  # dim1, 0
                   d1_arr[-1] + d1_arr[1],     # dim1, end + dx
                   d0_arr[-1] + d0_arr[1],     # dim0, end + dx
                   d0_arr[0]]                  # dim0, 0
    else:
        _extent = [0, elevation_data.shape[1],
                   elevation_data.shape[0], 0]

    # plot the data
    im = ax.imshow(
        elevation_data - datum,
        cmap=carto_cm, norm=carto_norm,
        extent=_extent, **kwargs)

    cb = append_colorbar(im, ax, **colorbar_kw)
    if not ticks:
        ax.set_xticks([], minor=[])
        ax.set_yticks([], minor=[])

    if return_im is True:
        return cb, im
    else:
        return cb


def overlay_sparse_array(sparse_array, ax=None, cmap='Reds',
                         alpha_clip=(None, 90), clip_type='percentile'):
    """Convenient plotting method to overlay a sparse 2D array on an image.

    Should only be used with data arrays that are sparse: i.e., where many
    elements take the value 0 (or very near 0).

    Original implementation was borrowed from the `dorado` project's
    implementation of `show_nourishment_area`.

    Parameters
    ----------
    sparse_array
        2d array to plot.

    ax : :obj:`~matplotlib.pyplot.Axes` object, optional
        A `matplotlib` `Axes` object to plot the section. Optional; if not
        provided, a call is made to ``plt.gca()`` to get the current (or
        create a new) `Axes` object.

    cmap
        Maplotlib colormap to use. String or colormap object.

    alpha_clip
        Two element tuple, specifying the *percentiles* of the data in
        `sparse_array` to clip for display. First element specifies the lower
        bound to clip, second element is the upper bound. Either or both
        elements can be `None` to indicate no clipping. Default is ``(None,
        90)``.

    clip_type
        String specifying how `alpha_clip` should be interpreted. Accepted
        values are `'percentile'` (default) and `'value'`. If `'percentile'`,
        the data in `sparse_array` are clipped based on the density of the
        data at the specified percentiles; note values should be in range
        [0, 100). If  `'value'`, the data in `sparse_array` are clipped
        directly based on the values in `alpha_clip`.

    Returns
    -------
    image
        image object returned from `imshow`.

    Examples
    --------
    Here, we use the discharge field from the model as an example
    of sparse data.

    .. plot::
        :include-source:
        :context: reset

        golfcube = dm.sample_data.golf()
        elevation_data = golfcube['eta'][-1, :, :]
        sparse_data = golfcube['discharge'][-1, ...]

        fig, ax = plt.subplots(1, 3, figsize=(8, 3))
        for axi in ax.ravel():
            dm.plot.aerial_view(elevation_data, ax=axi)

        dm.plot.overlay_sparse_array(
            sparse_data, ax=ax[0])  # default clip is (None, 90)
        dm.plot.overlay_sparse_array(
            sparse_data, alpha_clip=(None, None), ax=ax[1])
        dm.plot.overlay_sparse_array(
            sparse_data, alpha_clip=(70, 90), ax=ax[2])

        plt.tight_layout()
        plt.show()

    .. plot::
        :include-source:
        :context: close-figs

        fig, ax = plt.subplots(1, 3, figsize=(8, 3))
        for axi in ax.ravel():
            dm.plot.aerial_view(elevation_data, ax=axi)

        dm.plot.overlay_sparse_array(
            sparse_data, ax=ax[0],
            clip_type='value')  # default clip is (None, 90)
        dm.plot.overlay_sparse_array(
            sparse_data, ax=ax[1],
            alpha_clip=(None, 0.2), clip_type='value')
        dm.plot.overlay_sparse_array(
            sparse_data, ax=ax[2],
            alpha_clip=(0.4, 0.6), clip_type='value')

        plt.tight_layout()
        plt.show()

    """
    if not ax:
        fig, ax = plt.subplots()

    # check this is a tuple or list
    if isinstance(alpha_clip, tuple) or isinstance(alpha_clip, list):
        if len(alpha_clip) != 2:
            raise ValueError(
                '`alpha_clip` must be tuple or list of length 2.')
    else:  # if it is a tuple, check the length
        raise TypeError(
            '`alpha_clip` must be type `tuple`, '
            'but was type {0}.'.format(type(alpha_clip)))

    # check the clip_type flag
    clip_type_allow = ['percentile', 'value']
    if clip_type not in clip_type_allow:
        raise ValueError(
            'Bad value given for `clip_type` argument. Input argument must '
            'be one of `{0}`, but was `{1}`'.format(
                clip_type_allow, clip_type))

    # pull the cmap out
    if isinstance(cmap, str):
        cmap = plt.cm.get_cmap(cmap)
    else:
        cmap = cmap

    # get the extent to plot
    if isinstance(sparse_array, xr.core.dataarray.DataArray):
        d0, d1 = sparse_array.dims
        d0_arr, d1_arr = sparse_array[d0], sparse_array[d1]
        _extent = [d1_arr[0],                  # dim1, 0
                   d1_arr[-1] + d1_arr[1],     # dim1, end + dx
                   d0_arr[-1] + d0_arr[1],     # dim0, end + dx
                   d0_arr[0]]                  # dim0, 0
    else:
        _extent = [0, sparse_array.shape[1],
                   sparse_array.shape[0], 0]

    # process the clip field
    #  if first argument is given and percentile
    if (not (alpha_clip[0] is None)) and (clip_type == 'percentile'):
        amin = np.nanpercentile(sparse_array, alpha_clip[0])
    #  if first argument is given and value
    elif (not (alpha_clip[0] is None)) and (clip_type == 'value'):
        amin = alpha_clip[0]
    #  if first argument is not given
    else:
        amin = np.nanmin(sparse_array)
    #  if second argument is given and percentile
    if (not (alpha_clip[1] is None)) and (clip_type == 'percentile'):
        amax = np.nanpercentile(sparse_array, alpha_clip[1])
    #  if second argument is given and value
    elif (not (alpha_clip[1] is None)) and (clip_type == 'value'):
        amax = alpha_clip[1]
    #  if second argument is not given
    else:
        amax = np.nanmax(sparse_array)

    # normalize the alpha channel
    alphas = matplotlib.colors.Normalize(
        amin, amax, clip=True)(sparse_array)  # Normalize alphas

    # normalize the colors
    colors = matplotlib.colors.Normalize(
        np.nanmin(sparse_array),
        np.nanmax(sparse_array))(sparse_array)  # Normalize colors

    colors = cmap(colors)
    colors[..., -1] = alphas

    im = ax.imshow(
        colors, extent=_extent)

    return im
