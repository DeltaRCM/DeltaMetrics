import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from . import mask


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
