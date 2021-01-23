import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from . import mask


def compute_shoreline_roughness(shore_mask, land_mask, **kwargs):
    """Compute shoreline roughness.

    Computes the shoreline roughness metric:

    .. math::

        L_{shore} / \sqrt{A_{land}}

    given binary masks of the shoreline and land area.

    Parameters
    ----------
    shore_mask : :obj:`~deltametrics.mask.ShorelineMask`, :obj:`ndarray`
        Shoreline mask. Can be a :obj:`~deltametrics.mask.ShorelineMask` object,
        or a binarized array.

    land_mask : :obj:`~deltametrics.mask.LandMask`, :obj:`ndarray`
        Land mask. Can be a :obj:`~deltametrics.mask.LandMask` object,
        or a binarized array.

    Returns
    -------
    roughness : :obj:`float`
        Shoreline roughness, computed as described above.
    """
    # extract data from masks
    if isinstance(land_mask, mask.LandMask):
        _lm = land_mask.mask
    else:
        _lm = land_mask

    _ = kwargs.pop('return_line', None)  # trash this variable if passed
    shorelength, shoreline = compute_shoreline_length(
        shore_mask, return_line=True)

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

    
    """
    # check if mask or already array
    if isinstance(shore_mask, mask.ShorelineMask):
        _sm = shore_mask.mask
    else:
        _sm = shore_mask

    if not (np.sum(_sm) > 0):
        raise ValueError('No pixels in shoreline mask.')

    # find where the mask is True (all x-y pairs along shore)
    _y, _x = np.argwhere(_sm).T

    # preallocate line arrays
    line_xs = np.zeros(len(_x))
    line_ys = np.zeros(len(_y))

    # determine a starting coordinate based on the proximity to the origin
    _closest = np.argmin(
        np.sqrt((_x - origin[0])**2 + (_y - origin[1])**2))
    line_xs[0] = _x[_closest]
    line_ys[0] = _y[_closest]
    
    # preallocate an array to track whether a point has been used
    _hit = np.zeros(len(_x), dtype=np.bool)
    _hit[_closest] = True

    # loop through all of the other points and organize into a line
    for i in range(len(_x)-1):
        # compute distance from ith point to all other points
        _xi, _yi = line_xs[i], line_ys[i]
        _dists = np.sqrt((_x[~_hit]-_xi)**2 + (_y[~_hit]-_yi)**2)

        # find where the distance is minimized (i.e., next point)
        _whr = np.argmin(_dists)

        # fill the line array with that point
        line_xs[i+1] = _x[~_hit][_whr]
        line_ys[i+1] = _y[~_hit][_whr]

        # find that point in the hit list and update it
        __whr = np.argwhere(~_hit)
        _hit[__whr[_whr]] = True

    line = np.column_stack((line_xs, line_ys))
    length = np.sum(np.sqrt((line_xs[1:]-line_xs[:-1])**2 +
                            (line_ys[1:]-line_ys[:-1])**2))

    if return_line:
        return length, line
    else:
        return length

def a_land_function(mask):
    """Compute a land-water function

    This function does blah blah...

    Parameters
    ----------
    mask : land-waterMask
        A land-water mask instance.

    Returns
    -------
    float
        something who knows what, maybe has math: :math:`\\alpha`.
    """
    pass


def compute_delta_radius(mask):
    """Compute the delta radius

    This function does blah blah...

    Parameters
    ----------
    mask : land-waterMask
        A land-water mask instance.

    Returns
    -------
    float
        something who knows what, maybe has math: :math:`\\alpha`.
    """
    pass


def a_channel_function(mask):
    """Compute a channel function

    This function does blah blah...

    Parameters
    ----------
    mask : ChannelMask
        A channel mask instance.

    Returns
    -------
    float
        something who knows what, maybe has math: :math:`\\alpha`.
    """
    pass


def compute_shoreline_angles(mask, param2=False):
    """Compute shoreline angle.

    Computes some stuff according to:

    .. math::

        \\theta = 7 \\frac{\\rho}{10-\\tau}

    where :math:`\\rho` is something, :math:`\\tau` is another.

    Parameters
    ----------

    mask : :obj:`~deltametrics.land.LandMask`
        LandMask with shoreline to compute along.

    param2 : float, optional
        Something else? It's assumed to be false.


    Returns
    -------

    vol_conc : float
        Volumetric concentration.


    Examples
    --------

    >>> dm.plan.compute_shoreline_angles(True, True)
    0.54

    """
    return 0.54
