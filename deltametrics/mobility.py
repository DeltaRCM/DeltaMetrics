"""Functions for channel mobility metrics.

Current available mobility metrics include:
    - Dry fraction decay from Cazanacli et al 2002
    - Planform overlap from Wickert et al 2013
    - Reworking index from Wickert et al 2013
    - Channel abandonment from Liang et al 2016

Also included are functions to fit curves to the output from the mobility
functions, allowing for decay constants and timescales to be quantified.
"""

import numpy as np
from deltametrics import mask
import xarray as xr
from scipy.optimize import curve_fit


def check_inputs(chmap, basevalues, time_window, landmap=None):
    """
    Check the input variable values.

    Ensures compatibility with mobility functions. Tries to convert from some
    potential input types (xarray data arrays, deltametrics masks) if possible.

    Parameters
    ----------
    chmap : ndarray
        A t-x-y shaped array with binary channel maps.

    basevalues : list
        List of t indices to use as the base channel maps.

    time_window : int
        Number of time slices (t indices) to use as the time lag

    landmap : ndarray, optional
        A t-x-y shaped array with binary land maps. Or a x-y 2D array of a
        binary base mask that contains 1s in the region to be analyzed and
        0s elsewhere.

    """
    # check binary map types - pull out ndarray from mask or xarray if needed
    if isinstance(chmap, np.ndarray) is True:
        pass
    elif isinstance(chmap, mask.BaseMask) is True:
        chmap = chmap.mask
    elif isinstance(chmap, xr.core.dataarray.DataArray) is True:
        chmap = chmap.values
    else:
        raise TypeError('chmap data type not understood.')

    if landmap is not None:
        if isinstance(landmap, np.ndarray) is True:
            pass
        elif isinstance(landmap, mask.BaseMask) is True:
            landmap = landmap.mask
        elif isinstance(landmap, xr.core.dataarray.DataArray) is True:
            landmap = landmap.values
        else:
            raise TypeError('landmap data type not understood.')

    # check basevalues and time_window types
    if isinstance(basevalues, list) is False:
        try:
            basevalues = list(basevalues)
        except Exception:
            raise TypeError('basevalues was not a list or list()-able obj.')

    if isinstance(time_window, int) is False:
        try:
            time_window = int(time_window)
        except Exception:
            raise TypeError('time_window was not an int or int()-able obj.')

    # check map shapes (expect 3D t-x-y arrays of same size)
    if len(chmap.shape) != 3:
        raise ValueError('Shape of chmap not 3D (expect t-x-y).')
    if landmap is not None:
        if len(landmap.shape) != 3:
            try:
                tmp_landmap = np.empty(chmap.shape)
                for i in range(0,tmp_landmap.shape[0]):
                    tmp_landmap[i, :, :] = landmap
                landmap = tmp_landmap
            except Exception:
                raise ValueError('Landmp does not match chmap, nor could it be'
                                 ' cast into an array that would match.')

        if np.shape(chmap) != np.shape(landmap):
            raise ValueError('Shapes of chmap and landmap do not match.')

    # check that the combined basemap + timewindow does not exceed max t-index
    Kmax = np.max(basevalues) + time_window
    if Kmax > chmap.shape[0]:
        raise ValueError('Largest basevalue + time_window exceeds max time.')

    # return the sanitized variables
    return chmap, landmap, basevalues, time_window


def calc_chan_decay(chmap, landmap, basevalues, time_window):
    """
    Calculate channel decay (reduction in dry fraction).

    Uses a method similar to that in Cazanacli et al 2002 to measure the
    dry fraction of a delta over time. This requires providing an input channel
    map, an input land map, choosing a set of base maps to use, and a time
    lag/time window over which to do the analysis.

    Parameters
    ----------
    chmap : ndarray
        A t-x-y shaped array with binary channel maps.

    landmap : ndarray
        A t-x-y shaped array with binary land maps. Or an x-y shaped array
        with the binary region representing the fluvial surface over which
        the mobility metric should be computed.

    basevalues : list
        List of t indices to use as the base channel maps.

    time_window : int
        Number of time slices (t indices) to use as the time lag

    Returns
    -------
    dryfrac : ndarray
        len(basevalues) x time_window 2-D array with each row representing
        the dry fraction (aka number of pixels not visited by a channel) in
        reference to a given base value at a certain time lag (column)

    """
    # sanitize the inputs first
    chmap, landmap, basevalues, time_window = check_inputs(chmap,
                                                           basevalues,
                                                           time_window,
                                                           landmap)

    # initialize dry fraction array
    dryfrac = np.zeros((len(basevalues), time_window))

    # loop through basevalues
    for i in range(0, len(basevalues)):
        # pull the corresponding time index for the base value
        Nbase = basevalues[i]
        # first get the dry map (non-channelized locations)
        base_dry = np.abs(chmap[Nbase, :, :] - landmap[Nbase, :, :])

        # define base landmap
        base_map = landmap[Nbase, :, :]

        # set first dry fraction at t=0
        dryfrac[i, 0] = np.sum(base_dry) / np.sum(base_map)

        # loop through the other maps to see how the dry area declines
        for Nstep in range(1, time_window):
            # get the incremental map
            chA_step = chmap[Nbase+Nstep, :, :]
            # subtract incremental map from dry map to get the new dry fraction
            base_dry -= chA_step
            # just want binary (1 = never channlized, 0 = channel visited)
            base_dry[base_dry < 0] = 0  # no need to have negative values
            # store remaining dry fraction
            dryfrac[i, Nstep] = np.sum(base_dry) / np.sum(base_map)

    return dryfrac


def calc_planform_overlap(chmap, landmap, basevalues, time_window):
    """
    Calculate channel planform overlap.

    Uses a method similar to that described in Wickert et al 2013 to measure
    the loss of channel system overlap with previous channel patterns. This
    requires an input channel map, land map, as well as defining the base maps
    to use and the time window over which you want to look.

    Parameters
    ----------
    chmap : ndarray
        A t-x-y shaped array with binary channel maps

    landmap : ndarray
        A t-x-y shaped array with binary land maps. Or an x-y shaped array
        with the binary region representing the fluvial surface over which
        the mobility metric should be computed.

    basevalues : list
        A list of values (t indices) to use for the base channel maps

    time_window : int
        Number of time slices (t values) to use for the transient maps

    Returns
    -------
    Ophi : ndarray
        A 2-D array of the normalized overlap values, array is of shape
        len(basevalues) x time_window so each row in the array represents
        the overlap values associated with a given base value and the
        columns are associated with each time lag.

    """
    # sanitize the inputs first
    chmap, landmap, basevalues, time_window = check_inputs(chmap,
                                                           basevalues,
                                                           time_window,
                                                           landmap)

    # initialize D, phi and Ophi
    D = np.zeros((len(basevalues), time_window))
    phi = np.zeros_like(D)
    Ophi = np.zeros_like(D)

    # loop through the base maps
    for j in range(0, len(basevalues)):
        # define variables associated with the base map
        fdrybase = 1 - (np.sum(chmap[basevalues[j], :, :]) /
                        np.sum(landmap[basevalues[j], :, :]))
        fwetbase = np.sum(chmap[basevalues[j], :, :]) / \
            np.sum(landmap[basevalues[j], :, :])
        # transient maps compared over the fluvial surface present in base map
        mask_map = landmap[basevalues[j], :, :]

        # loop through the transient maps associated with this base map
        for i in range(0, time_window):
            D[j, i] = np.sum(np.abs(chmap[basevalues[j], :, :]*mask_map -
                                    chmap[basevalues[j]+i, :, :]*mask_map))
            fdrystep = 1 - (np.sum(chmap[basevalues[j]+i, :, :]*mask_map) /
                            np.sum(mask_map))
            fwetstep = np.sum(chmap[basevalues[j]+i, :, :]*mask_map) / \
                np.sum(mask_map)
            phi[j, i] = fwetbase*fdrystep + fdrybase*fwetstep
            # for Ophi use a standard area in denominator, we use base area
            Ophi[j, i] = 1 - D[j, i]/(np.sum(mask_map)*phi[j, i])
    # just return the Ophi
    return Ophi


def calc_reworking_fraction(chmap, landmap, basevalues, time_window):
    """
    Calculate the reworking fraction.

    Uses a method similar to that described in Wickert et al 2013 to measure
    the reworking of the fluvial surface with time. This requires an input
    channel map, land map, as well as defining the base maps to use and the
    time window over which you want to look.

    Parameters
    ----------
    chmap : ndarray
        A t-x-y shaped array with binary channel maps

    landmap : ndarray
        A t-x-y shaped array with binary land maps. Or an x-y shaped array
        with the binary region representing the fluvial surface over which
        the mobility metric should be computed.

    basevalues : list
        A list of values (t indices) to use for the base channel maps

    time_window : int
        Number of time slices (t values) to use for the transient maps

    Returns
    -------
    fr : ndarray
        A 2-D array of the reworked fraction values, array is of shape
        len(basevalues) x time_window so each row in the array represents
        the overlap values associated with a given base value and the
        columns are associated with each time lag.

    """
    # sanitize the inputs first
    chmap, landmap, basevalues, time_window = check_inputs(chmap,
                                                           basevalues,
                                                           time_window,
                                                           landmap)

    # initialize unreworked pixels (Nbt) and reworked fraction (fr)
    Nbt = np.zeros((len(basevalues), time_window))
    fr = np.zeros_like(Nbt)

    # loop through the base maps
    for j in range(0, len(basevalues)):
        # define variables associated with the base map
        base = basevalues[j]
        # fluvial surface is considered to be the one present in base map
        basemask = landmap[base, :, :]
        notland = len(np.where(basemask == 0)[0])  # number of not-land pixels
        basechannels = chmap[base, :, :]
        fbase = basemask - basechannels
        fdrybase = np.sum(fbase) / np.sum(basemask)

        # initialize channelmap series through time (kb) using initial chmap
        kb = np.copy(basechannels)

        # loop through the transient maps associated with this base map
        for i in range(0, time_window):
            # if i == 0 no reworking has happened yet
            if i == 0:
                Nbt[j, i] = fdrybase * np.sum(basemask)
                fr[j, i] = 0
            else:
                # otherwise situation is different

                # transient channel map withint base fluvial surface
                tmap = chmap[base+i, :, :]*basemask

                # add to kb
                kb += tmap

                # unreworked pixels are those channels have not visited
                # get this by finding all 0s left in kb and subtracting
                # the number of non-land pixels from base fluvial surface
                unvisited = len(np.where(kb == 0)[0])
                Nbt[j, i] = unvisited - notland
                fr[j, i] = 1 - (Nbt[j, i]/(np.sum(basemask)*fdrybase))

    # just return the reworked fraction
    return fr


def calc_chan_abandonment(chmap, basevalues, time_window):
    """
    Calculate channel abandonment.

    Measure the number of channelized pixels that are no longer channelized as
    a signature of channel mobility based on method in Liang et al 2016. This
    requires providing an input channel map, and setting parameters for the
    min/max values to compare to, and the time window over which the evaluation
    can be done.

    Parameters
    ----------
    chmap : ndarray
        A t-x-y shaped array with binary channel maps

    basevalues : list
        A list of values (t indices) to use for the base channel maps

    time_window : int
        Number of time slices (t values) to use for the transient maps

    Returns
    -------
    PwetA : ndarray
        A 2-D array of the abandoned fraction of the channel over the window of
        time. It is of shape len(basevalues) x time_window so each row
        represents the fraction of the channel that has been abandonded, and
        the columns are associated with each time lag.

    """
    # sanitize the inputs first
    chmap, landmap, basevalues, time_window = check_inputs(chmap,
                                                           basevalues,
                                                           time_window)
    # initialize values
    PwetA = np.zeros((len(basevalues), time_window))
    chA_base = np.zeros_like(chmap[0, :, :])
    chA_step = np.zeros_like(chmap[0, :, :])

    # loop through the basevalues
    for i in range(0, len(basevalues)):
        # first get the 'base' channel map that is being compared to
        Nbase = basevalues[i]
        chA_base = chmap[Nbase, :, :]
        # get total number of channel pixels in that map
        baseA = np.sum(chA_base)
        # loop through the other maps to be compared against the base map
        for Nstep in range(1, time_window):
            # get the incremental map
            chA_step = chmap[Nbase+Nstep, :, :]
            # get the number of pixels that were abandonded
            stepA = len(np.where(chA_base.flatten() > chA_step.flatten())[0])
            # store this number in the PwetA array for each transient map
            PwetA[i, Nstep] = stepA/baseA

    return PwetA


def mobility_curve_fit(mdata, fit='harmonic'):
    """
    Calculate curve fit for channel mobility data.

    Given some mobility data output from one of the mobility metrics, fit a
    curve to the average of that data. Several functional forms are available
    for the fitting: exponential, harmonic, and linear functions.

    Parameters
    ----------
    mdata : ndarray
        Mobility data, either already averaged or a 2D array of of shape
        len(basevalues) x time_window. Any output from one of the mobility
        functions is acceptable.

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
    if len(mdata.shape) == 2:
        mdata = np.mean(mdata, axis=0)

    # define x data
    xdata = np.array(range(0, len(mdata)))

    # do fit
    if fit == 'harmonic':
        func_harmonic = lambda x, a, b: a / (1 + b * x)
        popt, pcov = curve_fit(func_harmonic, xdata, mdata)
        yfit = func_harmonic(xdata, *popt)
    elif fit == 'exponential':
        func_exponential = lambda x, a, b, c: (a - b) * np.exp(-c * x) + b
        popt, pcov = curve_fit(func_exponential, xdata, mdata)
        yfit = func_exponential(xdata, *popt)
    elif fit == 'linear':
        func_linear = lambda x, a, b: a * x + b
        popt, pcov = curve_fit(func_linear, xdata, mdata)
        yfit = func_linear(xdata, *popt)

    perror = np.sqrt(np.diag(pcov))

    return yfit, pcov, perror
