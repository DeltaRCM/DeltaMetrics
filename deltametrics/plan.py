import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from . import mask


def compute_shoreline_roughness(shore_mask, land_mask):
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

    if isinstance(shore_mask, mask.ShorelineMask):
        _sm = shore_mask.mask
    if isinstance(land_mask, mask.LandMask):
        _lm = land_mask.mask

    if np.sum(_sm) > 0:
        # sort the shoreline mask into a shoreline "line"
        _y, _x = np.argwhere(_sm).T
        _closest = np.argmin(np.sqrt((_x-0)**2 + (_y-0)**2))
        _xs = np.zeros(len(_x))
        _ys = np.zeros(len(_y))
        _xs[0] = _x[_closest]
        _ys[0] = _y[_closest]
        _hit = np.zeros(len(_x), dtype=np.bool)
        _hit[_closest] = True
        for i in range(len(_x)-1):
            _xi = _xs[i]
            _yi = _ys[i]
            _dists = np.sqrt((_x[~_hit]-_xi)**2 + (_y[~_hit]-_yi)**2)
            _whr = np.argmin(_dists)
            _xs[i+1] = _x[~_hit][_whr]
            _ys[i+1] = _y[~_hit][_whr]
            __whr = np.argwhere(~_hit)
            _hit[__whr[_whr]] = True

        shore_len_pix = np.sum(np.sqrt((_xs[1:]-_xs[:-1])**2 +
                                       (_ys[1:]-_ys[:-1])**2))
        land_area_pix = np.sum(_lm)
        rough = shore_len_pix / np.sqrt(land_area_pix)
    else:
        rough = np.nan
    return rough


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
