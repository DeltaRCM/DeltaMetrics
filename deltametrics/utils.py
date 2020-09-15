import os
import sys

import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy import optimize

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


class NoStratigraphyError(AttributeError):
    """Error message for access when no stratigraphy.

    Parameters
    ----------
    obj : :obj:`str`
        Which object user tried to access.

    var : :obj:`str`, optional
        Which variable user tried to access. If provided, more information
        is given in the error message.

    Examples
    --------

    Without the optional `var` argument:

    .. doctest::

        >>> raise utils.NoStratigraphyError(rcm8cube) #doctest: +SKIP
        deltametrics.utils.NoStratigraphyError: 'DataCube' object
        has no preservation or stratigraphy information.

    With the `var` argument given as ``'strat_attr'``:

    .. doctest::

        >>> raise utils.NoStratigraphyError(rcm8cube, 'strat_attr') #doctest: +SKIP
        deltametrics.utils.NoStratigraphyError: 'DataCube' object
        has no attribute 'strat_attr'.
    """

    def __init__(self, obj, var=None):
        """Documented in class docstring."""
        if not (var is None):
            message = "'" + type(obj).__name__ + "'" + " object has no attribute " \
                      "'" + var + "'."
        else:
            message = "'" + type(obj).__name__ + "'" + " object has no preservation " \
                      "or stratigraphy information."
        super().__init__(message)

"""
    yields an exception with:

    .. code::

        deltametrics.utils.NoStratigraphyError: 'DataCube' object
        has no preservation or stratigraphy information.


    .. code::

        deltametrics.utils.NoStratigraphyError: 'DataCube' object
        has no attribute 'strat_attr'.


"""

def needs_stratigraphy(func):
    """Decorator for properties requiring stratigraphy.
    """
    def decorator(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AttributeError as e:
            raise NoStratigraphyError(e)
    return decorator


class AttributeChecker(object):
    """Mixin attribute checker class.

    Registers a method to check whether ``self`` has a given attribute. I.e.,
    the function works as ``hasattr(self, attr)``, where ``attr`` is the
    attribute of interest.

    The benefit of this method over ``hasattr`` is that this method optionally
    takes a list of arguments, and returns a well-formatted error message, to
    help explain which attribute is necessary for a given operation.
    """

    def _attribute_checker(self, checklist):
        """Check for attributes of self.

        Parameters
        ----------
        checklist : `list` of `str`, `str`
            List of attributes to check for existing in ``self``. If a string
            is provided, a single attribute defined by the string is checked
            for. Otherwise, a list of strings is expected, and each string is
            checked.

        .. note::
            This can be refactored to work as a decorator that takes the
            required list as arguments. This could be faster during runtime.
        """

        att_dict = {}
        if type(checklist) is list:
            pass
        elif type(checklist) is str:
            checklist = [checklist]
        else:
            raise TypeError('Checklist must be of type `list`,'
                            'but was type: %s' % type(checklist))

        for c, check in enumerate(checklist):
            has = getattr(self, check, None)
            if has is None:
                att_dict[check] = False
            else:
                att_dict[check] = True

        log_list = [value for value in att_dict.values()]
        log_form = [value for string, value in
                    zip(log_list, att_dict.keys()) if not string]
        if not all(log_list):
            raise RuntimeError('Required attribute(s) not assigned: '
                               + str(log_form))
        return att_dict


def curve_fit(data, fit='harmonic'):
    """
    Calculate curve fit given some data.

    Several functional forms are available for fitting: exponential, harmonic,
    and linear. The input `data` can be 1-D, or 2-D, if it is 2-D, the data
    will be averaged. The expected 2-D shape is (Y-Values, # Values) where the
    data you wish to have fit is in the first dimension, and the second
    dimension is of len(# Values).

    E.g. Given some mobility data output from one of the mobility metrics,
    fit a curve to the average of that data.

    Parameters
    ----------
    data : ndarray
        Data, either already averaged or a 2D array of of shape
        len(data values) x len(# values).

    fit : str, optional (default is 'harmonic')
        A string specifying the type of function to be fit. Options are as
        follows:
            - 'exponential' : (a - b) * np.exp(-c * x) + b
            - 'harmonic' : a / (1 + b * x)
            - 'linear' : a * x + b

    Returns
    -------
    yfit : ndarray
        y-values corresponding to the fitted function.

    pcov : ndarray
        Covariance associated with the fitted function parameters.

    perror : ndarray
        One standard deviation error for the parameters (from pcov)

    """
    avail_fits = ['exponential', 'harmonic', 'linear']
    if fit not in avail_fits:
        raise ValueError('Fit specified is not valid.')

    # average the mobility data if needed
    if len(data.shape) == 2:
        data = np.mean(data, axis=0)

    # define x data
    xdata = np.array(range(0, len(data)))

    # do fit
    if fit == 'harmonic':
        def func_harmonic(x, a, b): return a / (1 + b * x)
        popt, pcov = optimize.curve_fit(func_harmonic, xdata, data)
        yfit = func_harmonic(xdata, *popt)
    elif fit == 'exponential':
        def func_exponential(x, a, b, c): return (a - b) * np.exp(-c * x) + b
        popt, pcov = optimize.curve_fit(func_exponential, xdata, data)
        yfit = func_exponential(xdata, *popt)
    elif fit == 'linear':
        def func_linear(x, a, b): return a * x + b
        popt, pcov = optimize.curve_fit(func_linear, xdata, data)
        yfit = func_linear(xdata, *popt)

    perror = np.sqrt(np.diag(pcov))

    return yfit, pcov, perror
