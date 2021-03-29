import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from skimage import feature
from skimage import morphology
from skimage import measure

from numba import jit, njit

import abc
import warnings

from . import mask
from . import cube
from . import plot
from . import utils


class BasePlanform(abc.ABC):
    """Base planform object.

    Defines common attributes and methods of a planform object.

    This object should wrap around many of the functions available from
    :obj:`~deltametrics.mask` and :obj:`~deltametrics.mobility`.
    """

    def __init__(self, planform_type, *args, name=None):
        """
        Identify coordinates defining the planform.

        Parameters
        ----------
        CubeInstance : :obj:`~deltametrics.cube.Cube` subclass instance, optional
            Connect to this cube. No connection is made if cube is not
            provided.

        Notes
        -----

        If no arguments are passed, an empty planform not connected to any cube
        is returned. This cube will will need to be manually connected to have
        any functionality (via the :meth:`connect` method).
        """
        # begin unconnected
        self._x = None
        self._y = None
        self._shape = None
        self._variables = None
        self.cube = None

        self.planform_type = planform_type
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
        """Connect this Planform instance to a Cube instance.
        """
        if not issubclass(type(CubeInstance), cube.BaseCube):
            raise TypeError('Expected type is subclass of {_exptype}, '
                            'but received was {_gottype}.'.format(
                                _exptype=type(cube.BaseCube),
                                _gottype=type(CubeInstance)))
        self.cube = CubeInstance
        self._variables = self.cube.variables
        self.name = name  # use the setter to determine the _name

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, var):
        if (self._name is None):
            # _name is not yet set
            self._name = var or self.planform_type
        else:
            # _name is already set
            if not (var is None):
                warnings.warn(
                    UserWarning("`name` argument supplied to instantiated "
                                "`Planform` object. To change the name of "
                                "a Planform, you must set the attribute "
                                "directly with `plan._name = 'name'`."))
            # do nothing

    @property
    def shape(self):
        return self._shape


class OpeningAnglePlanform(BasePlanform):
    """Planform for handling the Shaw Opening Angle Method.


    """

    @staticmethod
    def from_arrays(*args):
        raise NotImplementedError

    @staticmethod
    def from_elevation_data(_arr, **kwargs):
        """Create the Opening Angle Planform from elevation data.

        This process creates an ElevationMask from the input elevation array,
        and proceeds to make the OAP from the below sea level mask.

        . todo:: finish docstring

        """
        # make a temporary mask
        _em = mask.ElevationMask(
            _arr, **kwargs)
        # extract value for omask
        _below_mask = ~(_em.mask)
        return OpeningAnglePlanform(_below_mask)

    @staticmethod
    def from_ElevationMask(ElevationMask):
        _below_mask = ~(ElevationMask.mask)

        return OpeningAnglePlanform(_below_mask)

    @staticmethod
    def from_mask(UnknownMask):
        if isinstance(UnknownMask, mask.ElevationMask):
            return OpeningAnglePlanform.from_ElevationMask(UnknownMask)
        else:
            raise TypeError('Must be type: ElevationMask.')

    def __init__(self, *args, **kwargs):
        """Init.

        EXPECTS A BINARY OCEAN MASK AS THE INPUT!

        .. todo:: needs docstring.

        """
        super().__init__('opening angle')
        self._shape = None
        self._sea_angles = None
        self._below_mask = None

        # check for inputs to return or proceed
        if (len(args) == 0):
            _allow_empty = kwargs.pop('allow_empty', False)
            if _allow_empty:
                # do nothing and return partially instantiated object
                return
            else:
                raise ValueError('Expected 1 input, got 0.')
        if not (len(args) == 1):
            raise ValueError('Expected 1 input, got %s.' % str(len(args)))

        # process the argument to the omask needed for Shaw OAM
        if utils.is_ndarray_or_xarray(args[0]):
            _arr = args[0]
            # check that is boolean or integer binary
            if (_arr.dtype == np.bool):
                _below_mask = _arr.astype(np.int)
            elif (_arr.dtype == np.int):
                if np.logical_or(_arr == 0, _arr == 1):
                    _below_mask = _arr
                else:
                    ValueError('Not all 0 and 1 ints.')
            else:
                raise TypeError('Not bool or int')
        else:
            # bad type supplied as argument
            raise TypeError('Invalid type for argument.')

        self._shape = _below_mask.shape

        self._compute_from_below_mask(_below_mask)

    def _compute_from_below_mask(self, below_mask, **kwargs):

        sea_angles = np.zeros(self._shape)

        if np.any(below_mask == 0):

            # pixels present in the mask
            shoreangles, seaangles = shaw_opening_angle_method(
                below_mask, **kwargs)

            # translate flat seaangles values to the shoreline image
            flat_inds = list(map(
                lambda x: np.ravel_multi_index(x, sea_angles.shape),
                seaangles[:2, :].T.astype(int)))
            sea_angles.flat[flat_inds] = seaangles[-1, :]

        # assign shore_image to the mask object with proper size
        # self._shore_angles = shoreangles
        self._sea_angles = sea_angles

        # properly assign the oceanmap to the self.below_mask
        self._below_mask = below_mask

    @property
    def sea_angles(self):
        """Maximum opening angle view of the sea from a pixel.
        """
        return self._sea_angles

    @property
    def below_mask(self):
        """Mask for below sea level pixels.

        This is the starting point for the Opening Angle Method solution.
        """
        return self._below_mask


def compute_shoreline_roughness(shore_mask, land_mask, **kwargs):
    """Compute shoreline roughness.

    Computes the shoreline roughness metric:

    .. math::

        L_{shore} / \\sqrt{A_{land}}

    given binary masks of the shoreline and land area. The length of the
    shoreline is computed internally with :obj:`compute_shoreline_length`.

    Parameters
    ----------
    shore_mask : :obj:`~deltametrics.mask.ShorelineMask`, :obj:`ndarray`
        Shoreline mask. Can be a :obj:`~deltametrics.mask.ShorelineMask` object,
        or a binarized array.

    land_mask : :obj:`~deltametrics.mask.LandMask`, :obj:`ndarray`
        Land mask. Can be a :obj:`~deltametrics.mask.LandMask` object,
        or a binarized array.

    **kwargs
        Keyword argument are passed to :obj:`compute_shoreline_length`
        internally.

    Returns
    -------
    roughness : :obj:`float`
        Shoreline roughness, computed as described above.

    Examples
    --------

    Compare the roughness of the shoreline early in the model simulation with
    the roughness later.

    .. plot::
        :include-source:

        rcm8 = dm.sample_data.cube.rcm8()

        # early in model run
        lm0 = dm.mask.LandMask(rcm8['eta'][5, :, :])
        sm0 = dm.mask.ShorelineMask(rcm8['eta'][5, :, :])

        # late in model run
        lm1 = dm.mask.LandMask(rcm8['eta'][-1, :, :])
        sm1 = dm.mask.ShorelineMask(rcm8['eta'][-1, :, :])

        # compute roughnesses
        rgh0 = dm.plan.compute_shoreline_roughness(sm0, lm0)
        rgh1 = dm.plan.compute_shoreline_roughness(sm1, lm1)

        # make the plot
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        rcm8.show_plan('eta', t=5, ax=ax[0])
        ax[0].set_title('roughness = {:.2f}'.format(rgh0))
        rcm8.show_plan('eta', t=-1, ax=ax[1])
        ax[1].set_title('roughness = {:.2f}'.format(rgh1))
        plt.show()
    """
    # extract data from masks
    if isinstance(land_mask, mask.LandMask):
        _lm = land_mask.mask
    else:
        _lm = land_mask

    _ = kwargs.pop('return_line', None)  # trash this variable if passed
    shorelength = compute_shoreline_length(
        shore_mask, return_line=False, **kwargs)

    # compute the length of the shoreline and area of land
    shore_len_pix = shorelength
    land_area_pix = np.sum(_lm)

    if (land_area_pix > 0):
        # compute roughness
        rough = shore_len_pix / np.sqrt(land_area_pix)
    else:
        raise ValueError('No pixels in land mask.')

    return rough


def compute_shoreline_length(shore_mask, origin=[0, 0], return_line=False):
    """Compute the length of a shoreline from a mask of the shoreline.

    Algorithm attempts to determine the sorted coordinates of the shoreline
    from a :obj:`~dm.mask.ShorelineMask`.

    .. warning::

        Imperfect algorithm, which may not include all `True` pixels in the
        `ShorelineMask` in the determined shoreline.

    Parameters
    ----------
    shore_mask : :obj:`~deltametrics.mask.ShorelineMask`, :obj:`ndarray`
        Shoreline mask. Can be a :obj:`~deltametrics.mask.ShorelineMask` object,
        or a binarized array.
    
    origin : :obj:`list`, :obj:`np.ndarray`, optional
        Determines the location from where the starting point of the line
        sorting is initialized. The starting point of the line is determined
        as the point nearest to `origin`. For non-standard data
        configurations, it may be important to set this to an appropriate
        value. Default is [0, 0].

    return_line : :obj:`bool`
        Whether to return the sorted line as a second argument. If True, a
        ``Nx2`` array of x-y points is returned. Default is `False`.

    Returns
    -------
    length : :obj:`float`
        Shoreline length, computed as described above.

    line : :obj:`np.ndarray`
        If :obj:`return_line` is `True`, the shoreline, as an ``Nx2`` array of
        x-y points, is returned.

    Examples
    --------

    Compare the length of the shoreline early in the model simulation with
    the length later.

    .. plot::
        :include-source:

        rcm8 = dm.sample_data.cube.rcm8()

        # early in model run
        sm0 = dm.mask.ShorelineMask(rcm8['eta'][5, :, :])

        # late in model run
        sm1 = dm.mask.ShorelineMask(rcm8['eta'][-1, :, :])

        # compute lengths
        len0 = dm.plan.compute_shoreline_length(sm0)
        len1, line1 = dm.plan.compute_shoreline_length(sm1, return_line=True)

        # make the plot
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        rcm8.show_plan('eta', t=5, ax=ax[0])
        ax[0].set_title('length = {:.2f}'.format(len0))
        rcm8.show_plan('eta', t=-1, ax=ax[1])
        ax[1].plot(line1[:, 0], line1[:, 1], 'r-')
        ax[1].set_title('length = {:.2f}'.format(len1))
        plt.show()
    """
    # check if mask or already array
    if isinstance(shore_mask, mask.ShorelineMask):
        _sm = shore_mask.mask
    else:
        _sm = shore_mask

    if not (np.sum(_sm) > 0):
        raise ValueError('No pixels in shoreline mask.')

    if _sm.ndim == 3:
        _sm = _sm.squeeze()

    # find where the mask is True (all x-y pairs along shore)
    _y, _x = np.argwhere(_sm).T

    # preallocate line arrays
    line_xs_0 = np.zeros(len(_x),)
    line_ys_0 = np.zeros(len(_y),)

    # determine a starting coordinate based on the proximity to the origin
    _closest = np.argmin(
        np.sqrt((_x - origin[0])**2 + (_y - origin[1])**2))
    line_xs_0[0] = _x[_closest]
    line_ys_0[0] = _y[_closest]
    
    # preallocate an array to track whether a point has been used
    hit_pts = np.zeros(len(_x), dtype=np.bool)
    hit_pts[_closest] = True

    # compute the distance to the next point
    dists_pts = np.sqrt((_x[~hit_pts]-_x[_closest])**2 + (_y[~hit_pts]-_y[_closest])**2)
    dist_next = np.min(dists_pts)
    dist_max = np.sqrt(15)

    # # loop through all of the other points and organize into a line
    idx = 0
    while (dist_next <= dist_max):
        
        idx += 1

        # find where the distance is minimized (i.e., next point)
        _whr = np.argmin(dists_pts)

        # fill the line array with that point
        line_xs_0[idx] = _x[~hit_pts][_whr]
        line_ys_0[idx] = _y[~hit_pts][_whr]

        # find that point in the hit list and update it
        __whr = np.argwhere(~hit_pts)
        hit_pts[__whr[_whr]] = True

        # compute distance from ith point to all other points
        _xi, _yi = line_xs_0[idx], line_ys_0[idx]
        dists_pts = np.sqrt((_x[~hit_pts]-_xi)**2 + (_y[~hit_pts]-_yi)**2)
        if (not np.all(hit_pts)):
            dist_next = np.min(dists_pts)
        else:
            dist_next = np.inf

    # trim the list
    line_xs_0 = np.copy(line_xs_0[:idx+1])
    line_ys_0 = np.copy(line_ys_0[:idx+1])

    #############################################
    # return to the first point and iterate again
    line_xs_1 = np.zeros(len(_x),)
    line_ys_1 = np.zeros(len(_y),)

    if (not np.all(hit_pts)):

        # compute dists from the intial point
        dists_pts = np.sqrt((_x[~hit_pts]-line_xs_0[0])**2 + (_y[~hit_pts]-line_ys_0[0])**2)
        dist_next = np.min(dists_pts)

        # loop through all of the other points and organize into a line
        idx = -1
        while (dist_next <= dist_max):
            
            idx += 1

            # find where the distance is minimized (i.e., next point)
            _whr = np.argmin(dists_pts)

            # fill the line array with that point
            line_xs_1[idx] = _x[~hit_pts][_whr]
            line_ys_1[idx] = _y[~hit_pts][_whr]

            # find that point in the hit list and update it
            __whr = np.argwhere(~hit_pts)
            hit_pts[__whr[_whr]] = True

            # compute distance from ith point to all other points
            _xi, _yi = line_xs_1[idx], line_ys_1[idx]
            dists_pts = np.sqrt((_x[~hit_pts]-_xi)**2 + (_y[~hit_pts]-_yi)**2)
            if (not np.all(hit_pts)):
                dist_next = np.min(dists_pts)
            else:
                dist_next = np.inf

        # trim the list
        line_xs_1 = np.copy(line_xs_1[:idx+1])
        line_ys_1 = np.copy(line_ys_1[:idx+1])
    else:
        line_xs_1 = np.array([])
        line_ys_1 = np.array([])

    # combine the lists
    line_xs = np.hstack((np.flip(line_xs_1), line_xs_0))
    line_ys = np.hstack((np.flip(line_ys_1), line_ys_0))

    # combine the xs and ys
    line = np.column_stack((line_xs, line_ys))
    length = np.sum(np.sqrt((line_xs[1:]-line_xs[:-1])**2 + (line_ys[1:]-line_ys[:-1])**2))

    if return_line:
        return length, line
    else:
        return length


@njit
def _compute_angles_between(c1, shoreandborder, Shallowsea, numviews):
    maxtheta = np.zeros((numviews, c1))
    for i in range(c1):

        shallow_reshape = np.atleast_2d(Shallowsea[:, i]).T
        diff = shoreandborder - shallow_reshape
        x = diff[0]
        y = diff[1]

        angles = np.arctan2(x, y)
        angles = np.sort(angles) * 180. / np.pi

        dangles = np.zeros_like(angles)
        dangles[:-1] = angles[1:] - angles[:-1]
        remangle = 360 - (angles.max() - angles.min())
        dangles[-1] = remangle
        dangles = np.sort(dangles)

        maxtheta[:, i] = dangles[-numviews:]

    return maxtheta


def shaw_opening_angle_method(below_mask, numviews=3):
    """Extract the opening angle map from an image.

    Applies the opening angle method [1]_ to compute the shoreline mask.
    Adapted from the Matlab implementation in [2]_.

    This *function* takes an image and extracts its opening angle map.

    .. [1] Shaw, John B., et al. "An image‐based method for
       shoreline mapping on complex coasts." Geophysical Research Letters
       35.12 (2008).

    .. [2] Liang, Man, Corey Van Dyk, and Paola Passalacqua.
       "Quantifying the patterns and dynamics of river deltas under
       conditions of steady forcing and relative sea level rise." Journal
       of Geophysical Research: Earth Surface 121.2 (2016): 465-496.

    Parameters
    ----------
    below_mask : ndarray
        Binary image that has been thresholded to split water/land. At
        minimum, this should be a thresholded elevation matrix, or some
        classification of land/water based on pixel color or reflectance
        intensity. This is the startin point (i.e., guess) for the opening
        angle method.

    numviews : int
        Defines the number of times to 'look' for the opening angle map.
        Default is 3.

    Returns
    -------
    shoreangles : ndarray
        Flattened values corresponding to the shoreangle detected for each
        'look' of the opening angle method

    seaangles : ndarray
        Flattened values corresponding to the 'sea' angle detected for each
        'look' of the opening angle method. The 'sea' region is the convex
        hull which envelops the shoreline as well as the delta interior.
    """

    Sx, Sy = np.gradient(below_mask)
    G = np.sqrt((Sx*Sx) + (Sy*Sy))

    # threshold the gradient to produce edges
    edges = np.logical_and((G > 0), (below_mask > 0))

    if np.sum(edges) == 0:
        raise ValueError(
            'No pixels identified in below_mask. '
            'Cannot compute the Opening Angle Method.')

    # extract coordinates of the edge pixels and define convex hull
    bordermap = np.pad(np.zeros_like(edges), 1, 'edge')
    bordermap[:-2, 1:-1] = edges
    bordermap[0, :] = 1
    points = np.fliplr(np.array(np.where(edges > 0)).T)
    hull = ConvexHull(points, qhull_options='Qc')

    # identify set of points to evaluate
    sea = np.fliplr(np.array(np.where(below_mask > 0.5)).T)

    # identify set of points in both the convex hull polygon and
    #   defined as points_to_test and put these binary points into seamap
    polygon = Polygon(points[hull.vertices]).buffer(0.01)
    In = utils._points_in_polygon(sea, np.array(polygon.exterior.coords))
    In = In.astype(np.bool)

    Shallowsea_ = sea[In]
    seamap = np.zeros(bordermap.shape)
    flat_inds = list(map(lambda x: np.ravel_multi_index(x, seamap.shape),
                         np.fliplr(Shallowsea_)))
    seamap.flat[flat_inds] = 1
    seamap[:3, :] = 0

    # define other points as these 'Deepsea' points
    Deepsea_ = sea[~In]
    Deepsea = np.zeros((numviews+2, len(Deepsea_)))
    Deepsea[:2, :] = np.flipud(Deepsea_.T)
    Deepsea[-1, :] = 200.  # 200 is a background value for waves1s later

    # define points for the shallow sea and the shoreborder
    Shallowsea = np.array(np.where(seamap > 0.5))
    shoreandborder = np.array(np.where(bordermap > 0.5))
    c1 = len(Shallowsea[0])
    maxtheta = np.zeros((numviews, c1))

    # compute angle between each shallowsea and shoreborder point
    maxtheta = _compute_angles_between(c1, shoreandborder, Shallowsea, numviews)

    # set up arrays for tracking the shore points and  their angles
    allshore = np.array(np.where(edges > 0))
    c3 = len(allshore[0])
    maxthetashore = np.zeros((numviews, c3))

    # get angles between the shore points and shoreborder points
    maxthetashore = _compute_angles_between(c3, shoreandborder, allshore, numviews)

    # define the shoreangles and seaangles identified
    shoreangles = np.vstack([allshore, maxthetashore])
    seaangles = np.hstack([np.vstack([Shallowsea, maxtheta]), Deepsea])

    return shoreangles, seaangles
