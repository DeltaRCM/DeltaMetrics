import os
import sys

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors

from . import io


def _get_version():
    """
    Extract version number from single file, and make it availabe everywhere.
    """
    from . import _version
    return _version.__version__()


def format_number(number):
    integer = int(round(number, -1))
    string = "{:,}".format(integer)
    return(string)


def format_table(number):
    integer = (round(number, 1))
    string = str(integer)
    return(string)


# class KnownVariable:
#     """KnownVariable decorator.
#     """
#     def __init__(self, func):
#         self.func = func
#         self.name = str(self.__name__)


class Colorset(object):
    """A default set of colors for attributes.

    This makes it easy to have consistent plots.
    """

    def __init__(self, override_list=None):
        """Initialize the Colorset.

        Initialize the set with default colormaps. 

        ..note::
            It is expected that any attribute added to known_list has a valid
            property defined in this class.

        Parameters
        ----------
        override : dict, optional
            Dictionary defining variable-colormap sets. Dictionary value must
            be either a string (and then match defined colormaps in matplotlib
            or DeltaMetrics), a new matplotlib Colormap object, or an Mx3
            numpy array that can be coerced into a linear colormap.
        """
        self.known_list = io.known_variables()

        # self.override_list = override_list
        # self.variable_list = {**self.known_list, **self.override_list}
        for var in self.known_list:
            setattr(self, var, None)
        if override_list:
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
        if type(var) is dict:
            self._variable_list = var
        else:
            raise TypeError('Invalid type for "override_list".'
                            'Must be type dict, but was type: %s '
                            % type(self.override_list))

    def __getitem__(self, var):
        # return getattr(self, var)
        return self.__getattribute__(var)

    @property
    def net_to_gross(self):
        return self._net_to_gross

    @net_to_gross.setter
    def net_to_gross(self):
        """Net-to-gross default colormap.
        """
        oranges = cm.get_cmap('Oranges', 64)
        greys = cm.get_cmap('Greys_r', 64)
        whiteblack = cm.get_cmap('Greys', 2)
        combined = np.vstack((greys(np.linspace(0.3, 0.6, 2)),
                              oranges(np.linspace(0.2, 0.8, 6))))
        ntgcmap = colors.ListedColormap(combined, name='net_to_gross')
        ntgcmap.set_bad("white")
        self._net_to_gross = ntgcmap

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, var):
        if not var:
            self._x = cm.get_cmap('viridis', 64)
        else:
            self._x = var

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, var):
        if not var:
            self._y = cm.get_cmap('viridis', 64)
        else:
            self._y = var

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, var):
        if not var:
            self._time = cm.get_cmap('viridis', 64)
        else:
            self._time = var

    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, var):
        if not var:
            self._eta = cm.get_cmap('cividis', 64)
        else:
            self._eta = var

    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self, var):
        if not var:
            self._stage = cm.get_cmap('viridis', 64)
        else:
            self._stage = var

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, var):
        if not var:
            self._depth = cm.get_cmap('viridis', 64)
        else:
            self._depth = var

    @property
    def discharge(self):
        return self._discharge

    @discharge.setter
    def discharge(self, var):
        if not var:
            self._discharge = cm.get_cmap('winter', 64)
        else:
            self._discharge = var

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, var):
        if not var:
            self._velocity = cm.get_cmap('plasma', 64)
        else:
            self._velocity = var

    @property
    def strata_age(self):
        return self._strata_age

    @strata_age.setter
    def strata_age(self, var):
        if not var:
            self._strata_age = cm.get_cmap('viridis', 64)
        else:
            self._strata_age = var

    @property
    def strata_sand_frac(self):
        return self._strata_sand_frac

    @strata_sand_frac.setter
    def strata_sand_frac(self, var):
        if not var:
            sandfrac = colors.ListedColormap(
                ['saddlebrown', 'sienna', 'goldenrod', 'gold'])
            sandfrac.set_under('white')
            bn = colors.BoundaryNorm([0, 1], sandfrac.N)
            self._strata_sand_frac = sandfrac
        else:
            self._strata_sand_frac = var

    @property
    def strata_depth(self):
        return self._strata_depth

    @strata_depth.setter
    def strata_depth(self, var):
        if not var:
            self._strata_depth = cm.get_cmap('viridis', 64)
        else:
            self._strata_depth = var
