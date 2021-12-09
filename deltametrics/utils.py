import numpy as np
import xarray as xr
from scipy import optimize
import time
import datetime

from numba import njit


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

        >>> raise utils.NoStratigraphyError(golfcube) #doctest: +SKIP
        deltametrics.utils.NoStratigraphyError: 'DataCube' object
        has no preservation or stratigraphy information.

    With the `var` argument given as ``'strat_attr'``:

    .. doctest::

        >>> raise utils.NoStratigraphyError(golfcube, 'strat_attr') #doctest: +SKIP
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


def is_ndarray_or_xarray(data):
    """Check that data is numpy array or xarray data.
    """
    truth = (isinstance(data, xr.core.dataarray.DataArray) or
             isinstance(data, np.ndarray))
    return truth


def curve_fit(data, fit='harmonic'):
    """Calculate curve fit given some data.

    Several functional forms are available for fitting: exponential, harmonic,
    and linear. The input `data` can be 1-D, or 2-D, if it is 2-D, the data
    will be averaged. The expected 2-D shape is (Y-Values, # Values) where the
    data you wish to have fit is in the first dimension, and the second
    dimension is of ``len(# Values)``.

    E.g. Given some mobility data output from one of the mobility metrics,
    fit a curve to the average of that data.

    Parameters
    ----------
    data : :obj:`ndarray`
        Data, either already averaged or a 2D array of of shape
        len(data values) x len(# values).

    fit : :obj:`str`, optional
        A string specifying the type of function to be fit. Options are as
        follows:
          * `exponential`, which evaluates :code:`(a - b) * np.exp(-c * x) + b`
          * `harmonic`, (default) which evaluates :code:`a / (1 + b * x)`
          * `linear`, which evaluates :code:`a * x + b`

    Returns
    -------
    yfit : :obj:`ndarray`
        y-values corresponding to the fitted function.

    popt : :obj:`array`
        Optimal values for the parameters of the function. Number of
        parameters is dependent on the functional form used.

    pcov : :obj:`ndarray`
        Covariance associated with the fitted function parameters.

    perror : :obj:`ndarray`
        One standard deviation error for the parameters (from pcov).
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

    return yfit, popt, pcov, perror


def determine_land_width(data, land_width_input=None):
    """Determine the land width from a dataset.
    """
    if (land_width_input is None):
        # determine the land width if not supplied explicitly
        trim_idx = guess_land_width_from_land(data)
    else:
        trim_idx = land_width_input
    return trim_idx


def guess_land_width_from_land(land_col_0):
    """Guess the land width from bed elevations.

    Utility to help autodetermine the domain setup. This utility should be
    replaced when possible by pulling domain setup variables directly from the
    netCDF file.

    Algortihm works by finding the point where the bed elevation is *flat*
    (i.e., where there is undisturbed basin).
    """
    i = 0
    delt = 10
    while i < len(land_col_0) - 1 and delt != 0:
        delt = land_col_0[i] - land_col_0[i+1]
        i += 1
    trim_idx = i - 1  # assign the trimming index
    return trim_idx


def coordinates_to_segments(coordinates):
    """Transform coordinate [x, y] array into line segments.

    Parameters
    ----------
    coordinates : :obj:`ndarray`
        `[N, 2]` array with `(x, y)` coordinate pairs in rows.

    Returns
    -------
    segments
        `[(N-1), 2, 2` array of line segments with dimensions of
        `each segment x x-coordinates x y-coordinates`.
    """
    _x = np.vstack((coordinates[:-1, 0],
                    coordinates[1:, 0])).T.reshape(-1, 2, 1)
    _y = np.vstack((coordinates[:-1, 1],
                    coordinates[1:, 1])).T.reshape(-1, 2, 1)
    return np.concatenate([_x, _y], axis=2)


def segments_to_cells(segments):
    """Transform a line segment (or array of segments) into integer coords.

    Helper function to convert a path into cells. This is generally used for
    determining the path of a `Section`, but is also available to use on any
    regularly gridded array by metric computation functions.

    Type and shape checking is performed, then the path (which may be
    composed of multiple vertices) is converted to a single path of cells.
    """
    _c = []  # append lists, do not know length of cells a priori
    for s in np.arange(segments.shape[0]):
        _c.append(line_to_cells(segments[s, ...]))
    _c = np.hstack(_c).T
    return _c


def line_to_cells(*args):
    """Convert a line to cell coordinates along the line.

    Takes as input the line to determine coordinates along. A line defined by
    two Cartesian coordinate endpoints `p1` and `p2`, may be specified  in one
    of three ways:

    * a single `ndarray` with fields ``[[x0, y0], [x1, y1]]``
    * a set of two-tuples with fields ``(x0, y0), (x1, y1)``
    * a set of four floats ``x0, y0, x1, y1``

    Cell coordinates will be ordered according to the order of the arguments
    (i.e., as though walking along the line from `p0` to `p1`).

    Parameters
    ----------
    line : four :obj:`float`, two :obj:`tuple`, one :obj:`ndarray`
        The line, defined in one of the options described above.

    Returns
    -------
    x : :obj:`ndarray`
        x-coordinates of cells intersected with line.

    y : :obj:`ndarray`
        y-coordinates of cells intersected with line.

    Examples
    --------

    .. plot::
        :include-source:

        >>> from deltametrics.utils import line_to_cells
        >>> p0 = (1, 6)
        >>> p1 = (6, 3)
        >>> x, y = line_to_cells(p0, p1)

        >>> fig, ax = plt.subplots(figsize=(2, 2))
        >>> _arr = np.zeros((10, 10))
        >>> _arr[y, x] = 1
        >>> ax.imshow(_arr, cmap='gray')
        >>> ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-')
        >>> plt.show()
    """
    # process the args
    if len(args) == 1:
        x0, y0, x1, y1 = args[0].ravel()
    elif len(args) == 2:
        x0, y0 = args[0]
        x1, y1 = args[1]
    elif len(args) == 4:
        x0, y0, x1, y1 = args
    else:
        raise TypeError(
            'Length of input must be 1, 2, or 4 but got: {0}'.format(args))

    # process the line to cells
    if np.abs(y1 - y0) < np.abs(x1 - x0):
        # if the line is "shallow" (dy < dx)
        if x0 > x1:
            # if the line is trending down (III)
            x, y = _walk_line((x1, y1), (x0, y0))
            x, y = np.flip(x), np.flip(y)  # flip order
        else:
            # if the line is trending up (I)
            x, y = _walk_line((x0, y0), (x1, y1))
    else:
        # if the line is "steep" (dy >= dx)
        if y0 > y1:
            # if the line is trending down (IV)
            y, x = _walk_line((y1, x1), (y0, x0))
            x, y = np.flip(x), np.flip(y)  # flip order
        else:
            # if the line is trending up (II)
            y, x = _walk_line((y0, x0), (y1, x1))

    return x, y


def _walk_line(p0, p1):
    """Walk a line to determine cells along path.

    Inputs depend on the steepness and direction of the input line. For a
    shallow line, where dx > dy, the input tuples should be in the form of
    `(x, y)` pairs. In contrast, for steep lines (where `dx < dy`), the input
    tuples should be `(y, x)` pairs.

    Additionally, the order of the tuples should be given based on whether the
    line is trending upward or downward, with increasing `x`.

    .. note:: TODO: finish descriptions for quadrants

    .. note::

        `x` in the code may not be cartesian x. Depends on quadrant of the
        line.
    """
    # unpack the point tuples
    x0, y0 = p0
    x1, y1 = p1

    dx, dy = x1 - x0, y1 - y0
    yi = 1
    if dy < 0:
        yi = -1
        dy = -dy

    D = 2 * dy - dx
    x = np.arange(x0, x1 + 1, dtype=int).T
    y = np.zeros((len(x),), dtype=int)

    yy = y0
    for i in np.arange(len(x)):
        y[i] = yy
        if D > 0:
            yy = yy + yi
            D = D - 2 * dx

        D = D + 2 * dy

    # sort by major axis, and index the cells
    xI = np.argsort(x)
    x = x[xI]
    y = y[xI]

    return x, y


def circle_to_cells(origin, radius, remove_duplicates=True):
    """The x, y pixel coordinates of a circle

    Use the mid-point circle algorithm is used for computation
    (http://en.wikipedia.org/wiki/Midpoint_circle_algorithm).

    Compute in advance the number of points that will be generated by the
    algorithm, to pre-allocate the coordinates arrays. Also, this function
    ensures that sorted coordinates are returned.

    This implementation removes duplicate points. Optionally, specify
    `remove_duplicates=False` to keep duplicates and achieve faster execution.
    Note that keeping duplicates is fine for drawing a circle with colored
    pixels, but for slicing an array along the arc, we need to have only
    unique and sorted indices.

    Original implementation from Jean-Yves Tinevez.
    <jeanyves.tinevez@gmail.com> - Nov 2011 - Feb 2012
    """
    x0, y0 = origin

    # Compute first the number of points
    octant_size = int((np.sqrt(2) * (radius - 1) + 4) / 2)
    n_points = 4 * octant_size
    xc = np.zeros((n_points,), dtype=int)
    yc = np.zeros((n_points,), dtype=int)

    x = 0
    y = radius
    f = 1 - radius
    dx = 1
    dy = - 2 * radius

    # 7th octant -- driver
    xc[0 * octant_size] = x0 - y
    yc[0 * octant_size] = y0 + x
    # 8th octant
    xc[2 * octant_size - 1] = x0 - x
    yc[2 * octant_size - 1] = y0 + y
    # 1st octant
    xc[2 * octant_size] = x0 + x
    yc[2 * octant_size] = y0 + y
    # 2nd octant
    xc[4 * octant_size - 1] = x0 + y
    yc[4 * octant_size - 1] = y0 + x

    for i in np.arange(1, n_points / 4, dtype=int):
        # update x and y, follwing midpoint algo
        if f > 0:
            y = y - 1
            dy = dy + 2
            f = f + dy
        x = x + 1
        dx = dx + 2
        f = f + dx

        # 7th octant
        xc[i] = x0 - y
        yc[i] = y0 + x
        # 8th octant
        xc[2 * octant_size - i - 1] = x0 - x
        yc[2 * octant_size - i - 1] = y0 + y
        # 1st octant
        xc[2 * octant_size + i] = x0 + x
        yc[2 * octant_size + i] = y0 + y
        # 2nd octant
        xc[4 * octant_size - i - 1] = x0 + y
        yc[4 * octant_size - i - 1] = y0 + x

    # There may be some duplicate entries
    #     We loop through to remove duplicates. This is slow, but necessary in
    #     most of our applications. We have to use something custom, rather
    #     than np.unique() because we need to preserve the ordering of the
    #     octants.
    if remove_duplicates:
        xyc = np.column_stack((xc, yc))
        keep = np.ones((n_points,), dtype=bool)
        for i in np.arange(1, 4):
            prv = xyc[(i-1)*octant_size:i*octant_size, :]
            nxt = xyc[i*octant_size:(i+1)*octant_size, :]
            dupe = np.nonzero(np.all(prv == nxt[:, np.newaxis], axis=2))[0]
            keep[(i*octant_size)+dupe] = False
        xyc = xyc[keep]
        xc = xyc[:, 0]
        yc = xyc[:, 1]

    # limit to positive indices (no wrapping)
    _and = np.logical_and(xc >= 0, yc >= 0)
    xc = xc[_and]
    yc = yc[_and]

    return xc, yc


@njit()
def _point_in_polygon(x, y, polygon):
    """Perform ray tracing.

    Used internally in methods to determine whether point is inside a polygon.

    Examples
    --------

    TODO
    """
    n = len(polygon)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x, p1y = polygon[0]
    for i in range(n+1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


@njit()
def _points_in_polygon(points, polygon):
    npts = points.shape[0]
    inside = np.zeros((npts,))

    for i in np.arange(npts):
        inside[i] = _point_in_polygon(points[i, 0], points[i, 1], polygon)

    return inside


def runtime_from_log(logname):
    """Calculate the model runtime from a logfile.

    Uses the timestamps in a logfile to compute model runtime.

    .. important::

        This function was written to work with the log files output from
        `pyDeltaRCM`, it may work for other log files, if the start of each
        line is a formatted timestamp: ``%Y-%m-%d %H:%M:%S``. 

    Parameters
    ----------
    logname : :obj:`str:`
        Path to the model logfile that you wish to get the runtime for.

    Returns
    -------
    runtime : :obj:`float`
        Float of the model runtime in seconds.
    """
    with open(logname) as f:
        lines = f.readlines()
        start = lines[0][:19]
        t_start = time.strptime(start, '%Y-%m-%d %H:%M:%S')
        t1 = time.mktime(t_start)
        fin = lines[-1][:19]
        t_end = time.strptime(fin, '%Y-%m-%d %H:%M:%S')
        t2 = time.mktime(t_end)
        te = datetime.timedelta(seconds=t2-t1)
    return te.total_seconds()
