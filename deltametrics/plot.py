import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import mpl_toolkits.axes_grid1 as axtk

from . import io

# plotting utilities

# def register_VariableInfo(func):


class VariableInfo(object):

    def __init__(self, name, **kwargs):
        self._name = name

        self.cmap = kwargs.pop('cmap', cm.get_cmap('viridis', 10))
        self.label = kwargs.pop('label', None)
        self.norm = kwargs.pop('norm', None)
        self.vmin = kwargs.pop('vmin', None)
        self.vmax = kwargs.pop('vmax', None)

    @property
    def name(self):
        return self._name

    @property
    def cmap(self):
        return self._cmap

    @cmap.setter
    def cmap(self, var):
        if type(var) is str:
            raise NotImplementedError
        elif issubclass(type(var), colors.Colormap):
            self._cmap = var
        else:
            raise TypeError

    @property
    def label(self):
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
        return self._norm

    @norm.setter
    def norm(self, var):
        self._norm = var

    @property
    def vmin(self):
        return self._vmin

    @vmin.setter
    def vmin(self, var):
        self._vmin = var

    @property
    def vmax(self):
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
    """

    def __init__(self, override_list=None):
        """Initialize the VariableSet.

        Initialize the set with default colormaps.

        .. note::
            It is expected that any attribute added to known_list has a valid
            property defined in this class.

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
        override : :obj:`dict`, optional
            Dictionary defining variable-property sets. Dictionary value must
            be either a string (and then match defined colormaps in matplotlib
            or DeltaMetrics), a new matplotlib Colormap object, or an Mx3
            numpy array that can be coerced into a linear colormap.
        """
        _added_list = ['net_to_gross']
        self.known_list = io.known_variables() + io.known_coords() + _added_list

        # self.override_list = override_list
        # self.variable_list = {**self.known_list, **self.override_list}
        for var in self.known_list:
            setattr(self, var, None)
        if override_list:
            raise NotImplementedError
            if not type(override_list) is dict:
                raise TypeError('Invalid type for "override_list".'
                                'Must be type dict, but was type: %s ' %
                                type(self.override_list))
            for var in override_list:
                if var in self.known_list:
                    setattr(self, var, override_list[var])
                else:
                    setattr(self, var, override_list[var])

    @property
    def variable_list(self):
        return self._variable_list

    @variable_list.setter
    def variable_list(self, var):
        raise NotImplementedError
        if type(var) is dict:
            self._variable_list = var
        else:
            raise TypeError('Invalid type for "override_list".'
                            'Must be type dict, but was type: %s '
                            % type(self.override_list))

    def __getitem__(self, var):
        """Get the attribute.

        This enables accessing variables by a variable name, in evaluation,
        rather than explicit typing of variable names.
        """
        return self.__getattribute__(var)

    @property
    def net_to_gross(self):
        """Net-to-gross style.

        .. plot::

            vs = dm.plot.VariableSet()
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            fig, ax = plt.subplots(figsize=(4,0.5))
            ax.imshow(gradient, aspect='auto', cmap=vs.net_to_gross.cmap)
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()

        """
        return self._net_to_gross

    @net_to_gross.setter
    def net_to_gross(self, _):
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

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, _):
        self._x = VariableInfo('x')

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, _):
        self._y = VariableInfo('y')

    @property
    def time(self):
        """Temporal history style.

        .. plot::

            vs = dm.plot.VariableSet()
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            fig, ax = plt.subplots(figsize=(4,0.5))
            ax.imshow(gradient, aspect='auto', cmap=vs.time.cmap)
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()
        """
        return self._time

    @time.setter
    def time(self, _):
        self._time = VariableInfo('time')

    @property
    def eta(self):
        """Bed elevation style.

        .. plot::

            vs = dm.plot.VariableSet()
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            fig, ax = plt.subplots(figsize=(4,0.5))
            ax.imshow(gradient, aspect='auto', cmap=vs.eta.cmap)
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()
        """
        return self._eta

    @eta.setter
    def eta(self, _):
        cmap = cm.get_cmap('cividis', 64)
        self._eta = VariableInfo('eta', cmap=cmap)

    @property
    def stage(self):
        """Flow stage style.

        .. plot::

            vs = dm.plot.VariableSet()
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            fig, ax = plt.subplots(figsize=(4,0.5))
            ax.imshow(gradient, aspect='auto', cmap=vs.stage.cmap)
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()
        """
        return self._stage

    @stage.setter
    def stage(self, _):
        self._stage = VariableInfo('stage')

    @property
    def depth(self):
        """Flow depth style.

        .. plot::

            vs = dm.plot.VariableSet()
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            fig, ax = plt.subplots(figsize=(4,0.5))
            ax.imshow(gradient, aspect='auto', cmap=vs.depth.cmap)
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()
        """
        return self._depth

    @depth.setter
    def depth(self, _):
        cmap = cm.get_cmap('Blues', 64)
        self._depth = VariableInfo('depth', cmap=cmap)

    @property
    def discharge(self):
        """Flow discharge style.

        .. plot::

            vs = dm.plot.VariableSet()
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            fig, ax = plt.subplots(figsize=(4,0.5))
            ax.imshow(gradient, aspect='auto', cmap=vs.discharge.cmap)
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()
        """
        return self._discharge

    @discharge.setter
    def discharge(self, _):
        cmap = cm.get_cmap('winter', 64)
        self._discharge = VariableInfo('discharge', cmap=cmap)

    @property
    def velocity(self):
        """Flow velocity style.

        .. plot::

            vs = dm.plot.VariableSet()
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            fig, ax = plt.subplots(figsize=(4,0.5))
            ax.imshow(gradient, aspect='auto', cmap=vs.velocity.cmap)
            # plt.subplots_adjust(bottom = 0.5)
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()
        """
        return self._velocity

    @velocity.setter
    def velocity(self, _):
        cmap = cm.get_cmap('plasma', 64)
        self._velocity = VariableInfo('velocity', cmap=cmap)

    @property
    def strata_sand_frac(self):
        """Sand fraction style.

        .. plot::

            vs = dm.plot.VariableSet()
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            fig, ax = plt.subplots(figsize=(4,0.5))
            ax.imshow(gradient, aspect='auto', cmap=vs.strata_sand_frac.cmap)
            # plt.subplots_adjust(bottom = 0.5)
            ax.set_yticks([])
            ax.set_xticks([])
            plt.show()

        """
        return self._strata_sand_frac

    @strata_sand_frac.setter
    def strata_sand_frac(self, _):
        sandfrac = colors.ListedColormap(
            ['saddlebrown', 'sienna', 'goldenrod', 'gold'])
        sandfrac.set_under('saddlebrown')
        bn = colors.BoundaryNorm([1e-6, 1], sandfrac.N)
        self._strata_sand_frac = VariableInfo('strata_sand_frac',
                                              cmap=sandfrac,
                                              norm=None, vmin=0)

    @property
    def strata_depth(self):
        return self._strata_depth

    @strata_depth.setter
    def strata_depth(self, _):
        self._strata_depth = VariableInfo('strata_depth')


def append_colorbar(ci, ax, **kwargs):
    """Append a colorbar, consistently placed.

    Parameters
    ----------
    ci : `matplotlib.pyplot.pcolormesh`, `matplotlib.pyplot.ImageAxes`
        The colored object generated via matplotlib that the colormap should
        be stolen from.

    ax : `matplotlib.Axes`
        The instance of axes to place the colorbar next to.

    Returns
    -------
    cb : `matplotlib.colorbar` instance.
        The colorbar instance created.
    """

    divider = axtk.axes_divider.make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.05)
    cb = plt.colorbar(ci, cax=cax)
    return cb


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
