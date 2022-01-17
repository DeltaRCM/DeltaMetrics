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


def check_inputs(chmap, basevalues=None, basevalues_idx=None, window=None,
                 window_idx=None, landmap=None):
    """
    Check the input variable values.

    Ensures compatibility with mobility functions. Tries to convert from some
    potential input types (xarray data arrays, deltametrics masks) if possible.

    Parameters
    ----------
    chmap : list, xarray.DataArray, numpy.ndarray
        Either a list of 2-D deltametrics.mask, xarray.DataArray, or
        numpy.ndarray objects, or a t-x-y 3-D xarray.DataArray or numpy.ndarray
        with channel mask values.

    basevalues : list, int, float, optional
        List of time values to use as the base channel map. (or single value)

    basevalues_idx : list, optional
        List of time indices to use as the base channel map. (or single value)

    window : int, float, optional
        Duration of time to use as the time lag (aka how far from the basemap
        will be analyzed).

    window_idx : int, float, optional
        Duration of time in terms of indices (# of save states) to use as the
        time lag.

    landmap : list, xarray.DataArray, numpy.ndarray, optional
        Either a list of 2-D deltametrics.mask, xarray.DataArray, or
        numpy.ndarray objects, or a t-x-y 3-D xarray.DataArray or numpy.ndarray
        with land mask values.

    """
    # handle input maps - try to convert some expected types
    in_maps = {'chmap': chmap, 'landmap': landmap}
    out_maps = {'chmap': None, 'landmap': None}
    _skip = False
    for key in in_maps.keys():
        if in_maps[key] is None:
            _skip = True
        else:
            inmap = in_maps[key]
            _skip = False
        # first expect a list of objects and coerce them into xarray.dataarrays
        if (isinstance(inmap, list)) and (_skip is False):
            # depending on type convert to xarray.dataarray
            if isinstance(inmap[0], np.ndarray) is True:
                dims = ('time', 'x', 'y')  # assumes an ultimate t-x-y shape
                if len(np.shape(inmap[0])) != 2:
                    raise ValueError('Expected list to be of 2-D ndarrays.')
                coords = {'time': np.arange(1),
                          'x': np.arange(inmap[0].shape[0]),
                          'y': np.arange(inmap[0].shape[1])}
                _converted = [xr.DataArray(
                    data=np.reshape(inmap[i],
                                    (1, inmap[i].shape[0], inmap[i].shape[1])),
                    coords=coords, dims=dims)
                              for i in range(len(inmap))]
            elif issubclass(type(inmap[0]), mask.BaseMask) is True:
                _converted = [i.mask for i in inmap]
            elif isinstance(inmap[0], xr.DataArray) is True:
                _converted = inmap
            else:
                raise TypeError('Type of objects in the input list is not '
                                'a recognized type.')
            # convert the list of xr.DataArrays into a single 3-D one
            out_maps[key] = _converted[0]  # init the final 3-D DataArray
            for j in range(1, len(_converted)):
                # stack them up along the time array into a 3-D dataarray
                out_maps[key] = xr.concat(
                    (out_maps[key], _converted[j]), dim='time').astype(float)

        elif (isinstance(inmap, np.ndarray) is True) and \
          (len(inmap.shape) == 3) and (_skip is False):
            dims = ('time', 'x', 'y')  # assumes t-x-y orientation of array
            coords = {'time': np.arange(inmap.shape[0]),
                      'x': np.arange(inmap.shape[1]),
                      'y': np.arange(inmap.shape[2])}
            out_maps[key] = \
                xr.DataArray(
                    data=inmap, coords=coords, dims=dims).astype(float)

        elif (issubclass(type(inmap), mask.BaseMask) is True) and \
          (_skip is False):
            raise TypeError(
                'Cannot input a Mask directly to mobility metrics. '
                'Use a list-of-masks instead.')

        elif (isinstance(inmap, xr.core.dataarray.DataArray) is True) and \
          (len(inmap.shape) == 3) and (_skip is False):
            out_maps[key] = inmap.astype(float)

        elif _skip is False:
            raise TypeError('Input mask data type or format not understood.')

    # can't do this binary check for a list
    # if ((chmap == 0) | (chmap == 1)).all():
    #     pass
    # else:
    #     raise ValueError('chmap was not binary')

    # check basevalues and time_window types
    if (basevalues is not None):
        try:
            baselist = list(basevalues)
            # convert to indices of the time dimension
            basevalues = [np.argmin(
                np.abs(out_maps['chmap'].time.data - i))
                for i in baselist]
        except Exception:
            raise TypeError('basevalues was not a list or list-able obj.')

    if (basevalues_idx is not None):
        try:
            basevalues_idx = list(basevalues_idx)
        except Exception:
            raise TypeError('basevalues_idx was not a list or list-able obj.')

    if (basevalues is not None) and (basevalues_idx is not None):
        raise Warning(
            'basevalues and basevalues_idx supplied, using `basevalues`.')
        base_out = basevalues
    elif (basevalues is None) and (basevalues_idx is not None):
        base_out = basevalues_idx
    elif (basevalues is not None) and (basevalues_idx is None):
        base_out = basevalues
    else:
        raise ValueError('No basevalue or basevalue_idx supplied!')

    if (window is not None) and \
      (isinstance(window, int) is False) and \
      (isinstance(window, float) is False):
        raise TypeError('Input window type was not an integer or float.')
    elif (window is not None):
        # convert to index of the time dimension
        _basetime = np.min(out_maps['chmap'].time.data)  # baseline time
        _reltime = out_maps['chmap'].time.data - _basetime  # relative time
        window = int(np.argmin(np.abs(_reltime - window)) + 1)

    if (window_idx is not None) and \
      (isinstance(window_idx, int) is False) and \
      (isinstance(window_idx, float) is False):
        raise TypeError(
            'Input window_idx type was not an integer or float.')

    if (window is not None) and (window_idx is not None):
        raise Warning(
            'window and window_idx supplied, using `window`.')
        win_out = window
    elif (window is None) and (window_idx is not None):
        win_out = window_idx
    elif (window is not None) and (window_idx is None):
        win_out = window
    else:
        raise ValueError('No window or window_idx supplied!')

    # check map shapes align
    if out_maps['landmap'] is not None:
        if np.shape(out_maps['chmap']) != np.shape(out_maps['landmap']):
            raise ValueError('Shapes of chmap and landmap do not match.')

    # check that the combined basemap + timewindow does not exceed max t-index
    Kmax = np.max(base_out) + win_out
    if Kmax > out_maps['chmap'].shape[0]:
        raise ValueError('Largest basevalue + time_window exceeds max time.')

    # collect name of the first dimenstion (should be time assuming t-x-y)
    dim0 = out_maps['chmap'].dims[0]

    # return the sanitized variables
    return out_maps['chmap'], out_maps['landmap'], base_out, win_out, dim0


def calculate_channel_decay(chmap, landmap,
                            basevalues=None, basevalues_idx=None,
                            window=None, window_idx=None):
    """
    Calculate channel decay (reduction in dry fraction).

    Uses a method similar to that in Cazanacli et al 2002 to measure the
    dry fraction of a delta over time. This requires providing an input channel
    map, an input land map, choosing a set of base maps to use, and a time
    lag/time window over which to do the analysis.

    Parameters
    ----------
    chmap : list, xarray.DataArray, numpy.ndarray
        Either a list of 2-D deltametrics.mask, xarray.DataArray, or
        numpy.ndarray objects, or a t-x-y 3-D xarray.DataArray or numpy.ndarray
        with channel mask values.

    landmap : list, xarray.DataArray, numpy.ndarray
        Either a list of 2-D deltametrics.mask, xarray.DataArray, or
        numpy.ndarray objects, or a t-x-y 3-D xarray.DataArray or numpy.ndarray
        with land mask values.

    basevalues : list, int, float, optional
        List of time values to use as the base channel map. (or single value)

    basevalues_idx : list, optional
        List of time indices to use as the base channel map. (or single value)

    window : int, float, optional
        Duration of time to use as the time lag (aka how far from the basemap
        will be analyzed).

    window_idx : int, float, optional
        Duration of time in terms of indices (# of save states) to use as the
        time lag.

    Returns
    -------
    dryfrac : ndarray
        len(basevalues) x time_window 2-D array with each row representing
        the dry fraction (aka number of pixels not visited by a channel) in
        reference to a given base value at a certain time lag (column)

    """
    # sanitize the inputs first
    chmap, landmap, basevalues, time_window, dim0 = check_inputs(
        chmap, basevalues, basevalues_idx, window, window_idx, landmap)

    # initialize dry fraction array
    dims = ('base', dim0)  # base and time-lag dimensions
    coords = {'base': np.arange(len(basevalues)),
              dim0: chmap[dim0][:time_window].values}
    dryfrac = xr.DataArray(
        data=np.zeros((len(basevalues), time_window)),
        coords=coords, dims=dims)

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
            base_dry.values[base_dry.values < 0] = 0  # no need to have negative values
            # store remaining dry fraction
            dryfrac[i, Nstep] = np.sum(base_dry) / np.sum(base_map)

    return dryfrac


def calculate_planform_overlap(chmap, landmap,
                               basevalues=None, basevalues_idx=None,
                               window=None, window_idx=None):
    """
    Calculate channel planform overlap.

    Uses a method similar to that described in Wickert et al 2013 to measure
    the loss of channel system overlap with previous channel patterns. This
    requires an input channel map, land map, as well as defining the base maps
    to use and the time window over which you want to look.

    Parameters
    ----------
    chmap : list, xarray.DataArray, numpy.ndarray
        Either a list of 2-D deltametrics.mask, xarray.DataArray, or
        numpy.ndarray objects, or a t-x-y 3-D xarray.DataArray or numpy.ndarray
        with channel mask values.

    landmap : list, xarray.DataArray, numpy.ndarray
        Either a list of 2-D deltametrics.mask, xarray.DataArray, or
        numpy.ndarray objects, or a t-x-y 3-D xarray.DataArray or numpy.ndarray
        with land mask values.

    basevalues : list, int, float, optional
        List of time values to use as the base channel map. (or single value)

    basevalues_idx : list, optional
        List of time indices to use as the base channel map. (or single value)

    window : int, float, optional
        Duration of time to use as the time lag (aka how far from the basemap
        will be analyzed).

    window_idx : int, float, optional
        Duration of time in terms of indices (# of save states) to use as the
        time lag.

    Returns
    -------
    Ophi : ndarray
        A 2-D array of the normalized overlap values, array is of shape
        len(basevalues) x time_window so each row in the array represents
        the overlap values associated with a given base value and the
        columns are associated with each time lag.

    """
    # sanitize the inputs first
    chmap, landmap, basevalues, time_window, dim0 = check_inputs(
        chmap, basevalues, basevalues_idx, window, window_idx, landmap)

    # initialize D, phi and Ophi
    dims = ('base', dim0)  # base and time-lag dimensions
    coords = {'base': np.arange(len(basevalues)),
              dim0: chmap[dim0][:time_window].values}
    D = xr.DataArray(
        data=np.zeros((len(basevalues), time_window)),
        coords=coords, dims=dims)
    phi = xr.zeros_like(D)
    Ophi = xr.zeros_like(D)

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


def calculate_reworking_fraction(chmap, landmap,
                                 basevalues=None, basevalues_idx=None,
                                 window=None, window_idx=None):
    """
    Calculate the reworking fraction.

    Uses a method similar to that described in Wickert et al 2013 to measure
    the reworking of the fluvial surface with time. This requires an input
    channel map, land map, as well as defining the base maps to use and the
    time window over which you want to look.

    Parameters
    ----------
    chmap : list, xarray.DataArray, numpy.ndarray
        Either a list of 2-D deltametrics.mask, xarray.DataArray, or
        numpy.ndarray objects, or a t-x-y 3-D xarray.DataArray or numpy.ndarray
        with channel mask values.

    landmap : list, xarray.DataArray, numpy.ndarray
        Either a list of 2-D deltametrics.mask, xarray.DataArray, or
        numpy.ndarray objects, or a t-x-y 3-D xarray.DataArray or numpy.ndarray
        with land mask values.

    basevalues : list, int, float, optional
        List of time values to use as the base channel map. (or single value)

    basevalues_idx : list, optional
        List of time indices to use as the base channel map. (or single value)

    window : int, float, optional
        Duration of time to use as the time lag (aka how far from the basemap
        will be analyzed).

    window_idx : int, float, optional
        Duration of time in terms of indices (# of save states) to use as the
        time lag.

    Returns
    -------
    fr : ndarray
        A 2-D array of the reworked fraction values, array is of shape
        len(basevalues) x time_window so each row in the array represents
        the overlap values associated with a given base value and the
        columns are associated with each time lag.

    """
    # sanitize the inputs first
    chmap, landmap, basevalues, time_window, dim0 = check_inputs(
        chmap, basevalues, basevalues_idx, window, window_idx, landmap)

    # initialize unreworked pixels (Nbt) and reworked fraction (fr)
    dims = ('base', dim0)  # base and time-lag dimensions
    coords = {'base': np.arange(len(basevalues)),
              dim0: chmap[dim0][:time_window].values}
    Nbt = xr.DataArray(
        data=np.zeros((len(basevalues), time_window)),
        coords=coords, dims=dims)
    fr = xr.zeros_like(Nbt)

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


def calculate_channel_abandonment(chmap, basevalues=None, basevalues_idx=None,
                                  window=None, window_idx=None):
    """
    Calculate channel abandonment.

    Measure the number of channelized pixels that are no longer channelized as
    a signature of channel mobility based on method in Liang et al 2016. This
    requires providing an input channel map, and setting parameters for the
    min/max values to compare to, and the time window over which the evaluation
    can be done.

    Parameters
    ----------
    chmap : list, xarray.DataArray, numpy.ndarray
        Either a list of 2-D deltametrics.mask, xarray.DataArray, or
        numpy.ndarray objects, or a t-x-y 3-D xarray.DataArray or numpy.ndarray
        with channel mask values.

    basevalues : list, int, float, optional
        List of time values to use as the base channel map. (or single value)

    basevalues_idx : list, optional
        List of time indices to use as the base channel map. (or single value)

    window : int, float, optional
        Duration of time to use as the time lag (aka how far from the basemap
        will be analyzed).

    window_idx : int, float, optional
        Duration of time in terms of indices (# of save states) to use as the
        time lag.

    Returns
    -------
    PwetA : ndarray
        A 2-D array of the abandoned fraction of the channel over the window of
        time. It is of shape len(basevalues) x time_window so each row
        represents the fraction of the channel that has been abandonded, and
        the columns are associated with each time lag.

    """
    # sanitize the inputs first
    chmap, landmap, basevalues, time_window, dim0 = check_inputs(
        chmap, basevalues, basevalues_idx, window, window_idx)
    # initialize values
    dims = ('base', dim0)  # base and time-lag dimensions
    coords = {'base': np.arange(len(basevalues)),
              dim0: chmap[dim0][:time_window].values}
    PwetA = xr.DataArray(
        data=np.zeros((len(basevalues), time_window)),
        coords=coords, dims=dims)
    chA_base = xr.zeros_like(chmap[0, :, :])
    chA_step = xr.zeros_like(chmap[0, :, :])

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
            stepA = len(
                np.where(chA_base.values.flatten() >
                         chA_step.values.flatten())[0])
            # store this number in the PwetA array for each transient map
            PwetA[i, Nstep] = stepA/baseA

    return PwetA


def channel_presence(chmap):
    """
    Calculate the normalized channel presence at each pixel location.

    Measure the normalized fraction of time a given pixel is channelized,
    based on method in Liang et al 2016. This requires providing a 3-D input
    channel map (t-x-y).

    Parameters
    ----------
    chmap : ndarray
        A t-x-y shaped array with binary channel maps

    Returns
    -------
    channel_presence : ndarray
        A x-y shaped array with the normalized channel presence values.

    Examples
    --------

    .. plot::
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> (x, y) = np.shape(golfcube['eta'][-1, ...])
        >>> # calculate channel masks/presence over final 5 timesteps
        >>> chmap = np.zeros((5, x, y))  # initialize channel map
        >>> for i in np.arange(-5, 0):
        ...     chmap[i, ...] = dm.mask.ChannelMask(
        ...         golfcube['eta'][i, ...], golfcube['velocity'][i, ...],
        ...         elevation_threshold=0, flow_threshold=0).mask
        >>>
        >>> fig, ax = plt.subplots(1, 2)
        >>> golfcube.quick_show('eta', ax=ax[0])  # final delta
        >>> p = ax[1].imshow(dm.mobility.channel_presence(chmap), cmap='Blues')
        >>> dm.plot.append_colorbar(p, ax[1], label='Channelized Time')
        >>> plt.show()

    """
    tmp_chans = None  # instantiate
    if isinstance(chmap, mask.ChannelMask) is True:
        chans = chmap._mask
    elif isinstance(chmap, np.ndarray) is True:
        tmp_chans = chmap
    elif isinstance(chmap, xr.DataArray) is True:
        chans = chmap
    elif isinstance(chmap, list) is True:
        # convert to numpy.ndarray if possible
        if (isinstance(chmap[0], np.ndarray) is True) \
          or (isinstance(chmap[0], xr.DataArray) is True):
            # init empty array
            tmp_chans = np.zeros(
                (len(chmap), chmap[0].squeeze().shape[0],
                 chmap[0].squeeze().shape[1]))
            # populate it
            for i in range(len(chmap)):
                if isinstance(chmap[0], xr.DataArray) is True:
                    tmp_chans[i, ...] = chmap[i].data.squeeze()
                else:
                    tmp_chans[i, ...] = chmap[i].squeeze()
        elif issubclass(type(chmap[0]), mask.BaseMask) is True:
            tmp_chans = [i.mask for i in chmap]
            # convert list to ndarray
            chans = np.zeros(
                (len(tmp_chans), tmp_chans[0].shape[1], tmp_chans[0].shape[2]))
            for i in range(chans.shape[0]):
                chans[i, ...] = tmp_chans[i]
        else:
            raise ValueError('Invalid values in the supplied list.')
    else:
        raise TypeError('chmap data type not understood.')
    # if tmp_chans is a numpy.ndarray, dimensions are not known
    if isinstance(tmp_chans, np.ndarray):
        dims = ('time', 'x', 'y')  # assumes an ultimate t-x-y shape
        coords = {'time': np.arange(tmp_chans.shape[0]),
                  'x': np.arange(tmp_chans.shape[1]),
                  'y': np.arange(tmp_chans.shape[2])}
        chans = xr.DataArray(data=tmp_chans, coords=coords, dims=dims)
    # calculation of channel presence is actually very simple
    channel_presence = np.sum(chans, axis=0) / chans.shape[0]
    return channel_presence
