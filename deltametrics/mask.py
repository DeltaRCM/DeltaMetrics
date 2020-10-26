"""Classes and methods to create masks of planform features and attributes."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from skimage import feature
from skimage import morphology
from skimage import measure

from . import utils


class BaseMask(object):
    """Low-level base class to be inherited by all mask objects."""

    def __init__(self, mask_type, data):
        """Initialize the base mask attributes and methods."""
        self.mask_type = mask_type
        # try to convert to np.ndarray if data not provided in that form
        if type(data) is not np.ndarray:
            try:
                self.data = data.__array__()
            except Exception:
                raise TypeError('Input data type must be numpy.ndarray,'
                                'but was ' + str(type(data)))
        else:
            self.data = data

        # set data to be 3D even if it is not (assuming t-x-y)
        if len(np.shape(self.data)) == 3:
            pass
        elif len(np.shape(self.data)) == 2:
            self.data = np.reshape(self.data, [1,
                                               np.shape(self.data)[0],
                                               np.shape(self.data)[1]])
        else:
            raise ValueError('Input data shape was not 2-D nor 3-D')

        self._mask = np.zeros(self.data.shape)

    @property
    def data(self):
        """ndarray : Values of the mask object.

        In setter, we should sanitize the inputs (enforce range 0-1) and
        convert everything to uints for speed and size.
        """
        return self._data

    @data.setter
    def data(self, var):
        self._data = var

    @property
    def mask(self):
        """ndarray : Binary mask values.

        Read-only mask attribute.
        """
        return self._mask

    def show(self, t=-1, ax=None, **kwargs):
        """Show the mask.

        Parameters
        ----------
        t : int, optional
            Value of time slice or integer from first dimension in 3D (t-x-y)
            convention to display the mask from. Default is -1 so the 'final'
            mask in time is displayed if this argument is not supplied.

        Passes `**kwargs` to ``matplotlib.imshow``.
        """
        if not ax:
            ax = plt.gca()
        cmap = kwargs.pop('cmap', 'gray')
        if hasattr(self, 'mask') and np.sum(self.mask) > 0:
            ax.imshow(self.mask[t, :, :], cmap=cmap, **kwargs)
            ax.set_title('A ' + self.mask_type + ' mask')
        else:
            raise AttributeError('No mask computed and/or mask is all 0s.')
        plt.draw()


class OAM(object):
    """Class for methods related to the Opening Angle Method [1]_."""

    def __init__(self, mask_type, data):
        """Require the same inputs as the :obj:`BaseMask` class."""
        self.mask_type = mask_type
        self.data = data

    def compute_shoremask(self, angle_threshold=75, **kwargs):
        """Compute the shoreline mask.

        Applies the opening angle method [1]_ to compute the shoreline mask.
        Particular method has been translated from [2]_.

        .. [1] Shaw, John B., et al. "An image‐based method for
           shoreline mapping on complex coasts." Geophysical Research Letters
           35.12 (2008).

        .. [2] Liang, Man, Corey Van Dyk, and Paola Passalacqua.
           "Quantifying the patterns and dynamics of river deltas under
           conditions of steady forcing and relative sea level rise." Journal
           of Geophysical Research: Earth Surface 121.2 (2016): 465-496.

        Parameters
        ----------
        angle_threshold : int, optional
            Threshold opening angle used in the OAM. Default is 75 degrees.

        Other Parameters
        ----------------
        topo_threshold : float, optional
            Threshold depth to use for the OAM. Default is -0.5.

        numviews : int, optional
            Defines the number of times to 'look' for the OAM. Default is 3.

        """
        # pop the kwargs
        self.topo_threshold = kwargs.pop('topo_threshold', -0.5)
        self.numviews = kwargs.pop('numviews', 3)

        # write the parameters to self
        self.angle_threshold = angle_threshold

        # initialize arrays
        self.oceanmap = np.zeros_like(self.data)
        self.shore_image = np.zeros_like(self.data)

        # loop through the time dimension
        for tval in range(0, self.data.shape[0]):
            trim_idx = utils.guess_land_width_from_land(self.data[tval, :, 0])
            data_trim = self.data[tval, trim_idx:, :]
            # use topo_threshold to identify oceanmap
            omap = (data_trim < self.topo_threshold) * 1.
            # if all ocean, there is no shore to be found - do next t-slice
            if (omap == 1).all():
                pass
            else:
                # apply seaangles function
                _, seaangles = self.Seaangles_mod(self.numviews, omap)
                # translate flat seaangles values to the shoreline image
                shore_image = np.zeros_like(data_trim)
                flat_inds = list(map(lambda x: np.ravel_multi_index(x,
                                                            shore_image.shape),
                                     seaangles[:2, :].T.astype(int)))
                shore_image.flat[flat_inds] = seaangles[-1, :]
                # grab contour from seaangles corresponding to angle threshold
                cs = measure.find_contours(shore_image,
                                           np.float(self.angle_threshold))
                C = cs[0]
                # convert this extracted contour to the shoreline mask
                shoremap = np.zeros_like(data_trim)
                flat_inds = list(map(lambda x: np.ravel_multi_index(x,
                                                            shoremap.shape),
                                     np.round(C).astype(int)))
                shoremap.flat[flat_inds] = 1
                # write shoreline map out to data.mask
                self._mask[tval, trim_idx:, :] = shoremap
                # assign shore_image to the mask object with proper size
                self.shore_image[tval, trim_idx:, :] = shore_image
                # properly assign the oceanmap to the self.oceanmap
                self.oceanmap[tval, trim_idx:, :] = omap

    def Seaangles_mod(self, numviews, thresholdimg):
        """Extract the opening angle map from an image.

        Adapted from the Matlab implementation in [2]_. Takes an image
        and extracts its opening angle map.

        Parameters
        ----------
        numviews : int
            Defines the number of times to 'look' for the opening angle map.

        thresholdimg : ndarray
            Binary image that has been thresholded to split deep water/land.

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
        Sx, Sy = np.gradient(thresholdimg)
        G = (Sx**2 + Sy**2)**.5

        # threshold the gradient to produce edges
        edges = (G > 0) & (thresholdimg > 0)

        # extract coordinates of the edge pixels and define convex hull
        bordermap = np.pad(np.zeros_like(edges), 1, 'edge')
        bordermap[:-2, 1:-1] = edges
        bordermap[0, :] = 1
        points = np.fliplr(np.array(np.where(edges > 0)).T)
        hull = ConvexHull(points, qhull_options='Qc')
        polygon = Polygon(points[hull.vertices]).buffer(0.01)

        # identify set of points to evaluate
        sea = np.fliplr(np.array(np.where(thresholdimg > 0.5)).T)
        points_to_test = [Point(i[0], i[1]) for i in sea]

        # identify set of points in both the convex hull polygon and
        # defined as points_to_test and put these binary points into seamap
        In = np.array(list(map(lambda pt: polygon.contains(pt),
                               points_to_test)))
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
        for i in range(c1):

            diff = shoreandborder - Shallowsea[:, i, np.newaxis]
            x = diff[0]
            y = diff[1]

            angles = np.arctan2(x, y)
            angles = np.sort(angles) * 180. / np.pi

            dangles = angles[1:] - angles[:-1]
            dangles = np.concatenate((dangles,
                                      [360 - (angles.max() - angles.min())]))
            dangles = np.sort(dangles)

            maxtheta[:, i] = dangles[-numviews:]

        # set up arrays for tracking the shore points and  their angles
        allshore = np.array(np.where(edges > 0))
        c3 = len(allshore[0])
        maxthetashore = np.zeros((numviews, c3))

        # get angles between the shore points and shoreborder points
        for i in range(c3):

            diff = shoreandborder - allshore[:, i, np.newaxis]
            x = diff[0]
            y = diff[1]

            angles = np.arctan2(x, y)
            angles = np.sort(angles) * 180. / np.pi

            dangles = angles[1:] - angles[:-1]
            dangles = np.concatenate((dangles,
                                      [360 - (angles.max() - angles.min())]))
            dangles = np.sort(dangles)

            maxthetashore[:, i] = dangles[-numviews:]

        # define the shoreangles and seaangles identified
        shoreangles = np.vstack([allshore, maxthetashore])
        seaangles = np.hstack([np.vstack([Shallowsea, maxtheta]), Deepsea])

        return shoreangles, seaangles


class ChannelMask(BaseMask, OAM):
    """Identify a binary channel mask.

    A channel mask object, helps enforce valid masking of channels.

    Examples
    --------
    Initialize the channel mask
        >>> velocity = rcm8cube['velocity'][-1, :, :]
        >>> topo = rcm8cube['eta'][-1, :, :]
        >>> cmsk = dm.mask.ChannelMask(velocity, topo)

    And visualize the mask:
        >>> cmsk.show()

    .. plot:: mask/channelmask.py

    """

    def __init__(self,
                 velocity,
                 topo,
                 velocity_threshold=0.3,
                 angle_threshold=75,
                 is_mask=False,
                 **kwargs):
        """Initialize the ChannelMask.

        Intializing the channel mask requires a flow velocity field and an
        array of the delta topography.

        Parameters
        ----------
        velocity : ndarray
            The velocity array to be used for mask creation.

        topo : ndarray
            The model topography to be used for mask creation.

        velocity_threshold : float, optional
            Threshold velocity above which flow is considered 'channelized'.
            The default value is 0.3 m/s based on DeltaRCM default parameters
            for sediment transport.

        angle_threshold : int, optional
            Threshold opening angle used in the OAM. Default is 75 degrees.

        is_mask : bool, optional
            Whether the data in :obj:`arr` is already a binary mask. Default is
            False. This should be set to True, if you have already binarized
            the data yourself, using custom routines, and want to just store
            the data in the ChannelMask object.

        Other Parameters
        ----------------
        landmask : :obj:`LandMask`, optional
            A :obj:`LandMask` object with a defined binary land mask.
            If given, it will be used to help define the channel mask.

        wetmask : :obj:`WetMask`, optional
            A :obj:`WetMask` object with a defined binary wet mask.
            If given, the landmask attribute it contains will be used to
            determine the channel mask.

        kwargs : optional
            Keyword arguments for :obj:`compute_shoremask`.

        """
        super().__init__(mask_type='channel', data=topo)
        if type(velocity) is np.ndarray:
            self.velocity = velocity
        else:
            try:
                self.velocity = velocity.__array__()
            except Exception:
                raise TypeError('Input velocity parameter is invalid. Must be'
                                'a np.ndarray or a'
                                'deltametrics.cube.CubeVariable object but it'
                                'was a: ' + str(type(velocity)))

        # assign **kwargs to self incase there are masks
        for key, value in kwargs.items():
            setattr(self, key, value)

        # write parameter values to self so they don't get lost
        self.velocity_threshold = velocity_threshold
        self.angle_threshold = angle_threshold

        if is_mask is False:
            self.compute_channelmask(**kwargs)
        elif is_mask is True:
            self._mask = self.data
        else:
            raise TypeError('is_mask must be a `bool`,'
                            'but was: ' + str(type(is_mask)))

    def compute_channelmask(self, **kwargs):
        """Compute the ChannelMask.

        Either uses an existing landmask, or recomputes it to use with the flow
        velocity threshold to calculate a binary channelmask.

        Other Parameters
        ----------------
        kwargs : optional
            Keyword arguments for :obj:`compute_shoremask`.

        """
        # check if a 'landmask' is available
        if hasattr(self, 'landmask'):
            try:
                self.oceanmap = self.landmask.oceanmap
                self.landmask = self.landmask.mask
            except Exception:
                self.compute_shoremask(self.angle_threshold, **kwargs)
                self.landmask = (self.shore_image < self.angle_threshold) * 1
        elif hasattr(self, 'wetmask'):
            try:
                self.landmask = self.wetmask.landmask
            except Exception:
                self.compute_shoremask(self.angle_threshold, **kwargs)
                self.landmask = (self.shore_image < self.angle_threshold) * 1
        else:
            self.compute_shoremask(self.angle_threshold, **kwargs)
            self.landmask = (self.shore_image < self.angle_threshold) * 1

        # compute a flowmap of cells where flow exceeds the velocity threshold
        self.flowmap = (self.velocity > self.velocity_threshold) * 1.

        # calculate the channelmask as the cells exceeding the threshold
        # within the topset of the delta (ignoring flow in ocean)
        self._mask = np.minimum(self.landmask, self.flowmap)


class WetMask(BaseMask, OAM):
    """Compute the wet mask.

    A wet mask object, identifies all wet pixels on the delta topset. Starts
    with the land mask and then uses the topo_threshold defined for the
    shoreline computation to add the wet pixels on the topset back to the mask.

    If a land mask has already been computed, then it can be used to define the
    wet mask. Otherwise the wet mask can be computed from scratch.

    Examples
    --------
    Initialize the wet mask
        >>> arr = rcm8cube['eta'][-1, :, :]
        >>> wmsk = dm.mask.WetMask(arr)

    And visualize the mask:
        >>> wmsk.show()

    .. plot:: mask/wetmask.py

    """

    def __init__(self,
                 arr,
                 angle_threshold=75,
                 is_mask=False,
                 **kwargs):
        """Initialize the WetMask.

        Intializing the wet mask requires either a 2-D array of data, or it
        can be computed if a :obj:`LandMask` has been previously computed.

        Parameters
        ----------
        arr : ndarray
            The data array to make the mask from.

        angle_threshold : int, optional
            Threshold opening angle used in the OAM. Default is 75 degrees.

        is_mask : bool, optional
            Whether the data in :obj:`arr` is already a binary mask. Default is
            False. This should be set to True, if you have already binarized
            the data yourself, using custom routines, and want to just store
            the data in the WetMask object.

        Other Parameters
        ----------------
        landmask : :obj:`LandMask`, optional
            A :obj:`LandMask` object with a defined binary shoreline mask.
            If given, the :obj:`LandMask` object will be checked for the
            `shore_image` and `angle_threshold` attributes.

        kwargs : optional
            Keyword arguments for :obj:`compute_shoremask`.

        """
        super().__init__(mask_type='wet', data=arr)

        # assign **kwargs to self in case the landmask was passed
        for key, value in kwargs.items():
            setattr(self, key, value)

        # put angle_threshold in self
        self.angle_threshold = angle_threshold

        if is_mask is False:
            self.compute_wetmask(**kwargs)
        elif is_mask is True:
            self._mask = self.data
        else:
            raise TypeError('is_mask must be a `bool`,'
                            'but was: ' + str(type(is_mask)))

    def compute_wetmask(self, **kwargs):
        """Compute the WetMask.

        Either recomputes the landmask, or uses precomputed information from
        the landmask to create the wetmask.

        """
        # check if a 'landmask' was passed in
        if hasattr(self, 'landmask'):
            try:
                self.oceanmap = self.landmask.oceanmap
                self.landmask = self.landmask.mask
            except Exception:
                self.compute_shoremask(self.angle_threshold, **kwargs)
                self.landmask = (self.shore_image < self.angle_threshold) * 1
        else:
            self.compute_shoremask(self.angle_threshold, **kwargs)
            self.landmask = (self.shore_image < self.angle_threshold) * 1
        # set wet mask as data.mask
        self._mask = self.oceanmap * self.landmask


class LandMask(BaseMask, OAM):
    """Identify a binary mask of the delta topset.

    A land mask object, helps enforce valid masking of delta topset.

    If a shoreline mask has been computed, it can be used to help compute the
    land mask, otherwise it will be computed from scratch.

    Examples
    --------
    Initialize the mask.
        >>> arr = rcm8cube['eta'][-1, :, :]
        >>> lmsk = dm.mask.LandMask(arr)

    And visualize the mask:
        >>> lmsk.show()

    .. plot:: mask/landmask.py

    """

    def __init__(self,
                 arr,
                 angle_threshold=75,
                 is_mask=False,
                 **kwargs):
        """Initialize the LandMask.

        Intializing the land mask requires an array of data, should be
        two-dimensional. If a shoreline mask (:obj:`ShoreMask`) has been
        computed, then this can be used to define the land mask. Otherwise the
        necessary computations will occur from scratch.

        Parameters
        ----------
        arr : ndarray
            2-D topographic array to make the mask from.

        angle_threshold : int, optional
            Threshold opening angle used in the OAM. Default is 75 degrees.

        is_mask : bool, optional
            Whether the data in :obj:`arr` is already a binary mask. Default
            value is False. This should be set to True, if you have already
            binarized the data yourself, using custom routines, and want to
            just store the data in the LandMask object.

        Other Parameters
        ----------------
        shoremask : :obj:`ShoreMask`, optional
            A :obj:`ShoreMask` object with a defined binary shoreline mask.
            If given, the :obj:`ShoreMask` object will be checked for the
            `shore_image` and `angle_threshold` attributes.

        kwargs : optional
            Keyword arguments for :obj:`compute_shoremask`.

        """
        super().__init__(mask_type='land', data=arr)

        # assign **kwargs to self in event a mask was passed
        for key, value in kwargs.items():
            setattr(self, key, value)

        # set parameter value to self so it is kept
        self.angle_threshold = angle_threshold

        if not is_mask:
            self.compute_landmask(**kwargs)
        elif is_mask is True:
            self._mask = self.data
        else:
            raise TypeError('is_mask must be a `bool`,'
                            'but was: ' + str(type(is_mask)))

    def compute_landmask(self, **kwargs):
        """Compute the LandMask.

        Uses data from the shoreline computation or recomputes the shoreline
        using the opening angle method, then an angle threshold is used to
        identify the land.

        """
        # check if a `shoremask' was passed in
        if hasattr(self, 'shoremask'):
            try:
                self.shore_image = self.shoremask.shore_image
                self.angle_threshold = self.shoremask.angle_threshold
            except Exception:
                self.compute_shoremask(self.angle_threshold, **kwargs)
        else:
            self.compute_shoremask(self.angle_threshold, **kwargs)
        # set the land mask to data.mask
        self._mask = (self.shore_image < self.angle_threshold) * 1


class ShorelineMask(BaseMask, OAM):
    """Identify the shoreline as a binary mask.

    A shoreline mask object, provides a binary identification of shoreline
    pixels.

    Examples
    --------
    Initialize the mask.
        >>> arr = rcm8cube['eta'][-1, :, :]
        >>> shrmsk = dm.mask.ShorelineMask(arr)

    Visualize the mask:
        >>> shrmsk.show()

    .. plot:: mask/shoremask.py

    """

    def __init__(self,
                 arr,
                 angle_threshold=75,
                 is_mask=False,
                 **kwargs):
        """Initialize the ShorelineMask.

        Initializing the shoreline mask requires a 2-D array of data. The
        shoreline mask is computed using the opening angle method (OAM)
        described in [1]_. The default parameters are designed for
        DeltaRCM and follow the implementation described in [2]_.

        Parameters
        ----------
        arr : ndarray
            2-D topographic array to make the mask from.

        angle_threshold : int, optional
            Threshold opening angle used in the OAM. Default is 75 degrees.

        is_mask : bool, optional
            Whether the data in :obj:`arr` is already a binary mask. Default
            value is False. This should be set to True, if you have already
            binarized the data yourself, using custom routines, and want to
            just store the data in the ShorelineMask object.

        Other Parameters
        ----------------
        kwargs : optional
            Keyword arguments for :obj:`compute_shoremask`.

        """
        super().__init__(mask_type='shore', data=arr)

        if is_mask is False:
            self.compute_shoremask(angle_threshold, **kwargs)
        elif is_mask is True:
            self._mask = self.data
        else:
            raise TypeError('is_mask must be a `bool`,'
                            'but was: ' + str(type(is_mask)))


class EdgeMask(BaseMask, OAM):
    """Identify the land-water edges.

    An edge mask object, delineates the boundaries between land and water.

    Examples
    --------
    Initialize the edge mask
        >>> arr = rcm8cube['eta'][-1, :, :]
        >>> emsk = dm.mask.EdgeMask(arr)

    Visualize the mask:
        >>> emsk.show()

    .. plot:: mask/edgemask.py

    """

    def __init__(self,
                 arr,
                 angle_threshold=75,
                 is_mask=False,
                 **kwargs):
        """Initialize the EdgeMask.

        Initializing the edge mask requires either a 2-D array of topographic
        data, or it can be computed using the :obj:`LandMask` and the
        :obj:`WetMask`.

        Parameters
        ----------
        arr : ndarray
            The data array to make the mask from.

        angle_threshold : int, optional
            Threshold opening angle used in the OAM. Default is 75 degrees.

        is_mask : bool, optional
            Whether the data in :obj:`arr` is already a binary mask. Default
            value is False. This should be set to True, if you have already
            binarized the data yourself, using custom routines, and want to
            just store the data in the EdgeMask object.

        Other Parameters
        ----------------
        landmask : :obj:`LandMask`, optional
            A :obj:`LandMask` object with the land identified

        wetmask : :obj:`WetMask`, optional
            A :obj:`WetMask` object with the surface water identified

        kwargs : optional
            Keyword arguments for :obj:`compute_shoremask`.

        """
        super().__init__(mask_type='edge', data=arr)

        # assign **kwargs to self incase there are masks
        for key, value in kwargs.items():
            setattr(self, key, value)

        # write parameter to self so it is not lost
        self.angle_threshold = angle_threshold

        if is_mask is False:
            self.compute_edgemask(**kwargs)
        elif is_mask is True:
            self._mask = self.data
        else:
            raise TypeError('is_mask must be a `bool`,'
                            'but was: ' + str(type(is_mask)))

    def compute_edgemask(self, **kwargs):
        """Compute the EdgeMask.

        Either computes it from just the topographic array, or uses information
        from :obj:`LandMask` and :obj:`WetMask` to create the edge mask.

        """
        # if landmask and edgemask exist it is quick
        if hasattr(self, 'landmask') and hasattr(self, 'wetmask'):
            try:
                self.landmask = self.landmask.mask
                self.wetmask = self.wetmask.mask
            except Exception:
                self.compute_shoremask(self.angle_threshold, **kwargs)
                self.landmask = (self.shore_image < self.angle_threshold) * 1
                self.wetmask = self.oceanmap * self.landmask
        elif hasattr(self, 'wetmask'):
            try:
                self.wetmask = self.wetmask.mask
                self.landmask = self.wetmask.landmask
            except Exception:
                self.compute_shoremask(self.angle_threshold, **kwargs)
                self.landmask = (self.shore_image < self.angle_threshold) * 1
                self.wetmask = self.oceanmap * self.landmask
        else:
            self.compute_shoremask(self.angle_threshold, **kwargs)
            self.landmask = (self.shore_image < self.angle_threshold) * 1
            self.wetmask = self.oceanmap * self.landmask
        # compute the edge mask
        for i in range(0, np.shape(self._mask)[0]):
            self._mask[i, :, :] = np.maximum(0,
                                             feature.canny(self.wetmask[i,
                                                                        :,
                                                                        :])*1 -
                                             feature.canny(self.landmask[i,
                                                                         :,
                                                                         :])*1)


class CenterlineMask(BaseMask):
    """Identify channel centerline mask.

    A centerline mask object, provides the location of channel centerlines.

    Examples
    --------
    Initialize the centerline mask
        >>> channelmask = dm.mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
                                              rcm8cube['eta'][-1, :, :])
        >>> clmsk = dm.mask.CenterlineMask(channelmask)

    Visualize the mask
        >>> clmsk.show()

    .. plot:: mask/centerlinemask.py

    """

    def __init__(self,
                 channelmask,
                 is_mask=False,
                 method='skeletonize',
                 **kwargs):
        """Initialize the CenterlineMask.

        Initialization of the centerline mask object requires a 2-D channel
        mask (can be the :obj:`ChannelMask` object or a binary 2-D array).

        Parameters
        ----------
        channelmask : :obj:`ChannelMask` or ndarray
            The channel mask to derive the centerlines from

        is_mask : bool, optional
            Whether the data in :obj:`arr` is already a binary mask. Default
            value is False. This should be set to True, if you have already
            binarized the data yourself, using custom routines, and want to
            just store the data in the CenterlineMask object.

        method : str, optional
            The method to use for the centerline mask computation. The default
            method ('skeletonize') is a morphological skeletonization of the
            channel mask.

        Other Parameters
        ----------------
        kwargs : optional
            Keyword arguments for the 'rivamap' functionality.

        """
        if isinstance(channelmask, np.ndarray):
            super().__init__(mask_type='centerline', data=channelmask)
        elif isinstance(channelmask, ChannelMask):
            super().__init__(mask_type='centerline',
                             data=channelmask.mask)
        else:
            raise TypeError('Input channelmask parameter is invalid. Must be'
                            'a np.ndarray or ChannelMask object but it was'
                            'a: ' + str(type(channelmask)))

        # save method type value to self
        self.method = method

        if is_mask is False:
            self.compute_centerlinemask(**kwargs)
        elif is_mask is True:
            self._mask = self.data
        else:
            raise TypeError('is_mask must be a `bool`,'
                            'but was: ' + str(type(is_mask)))

    def compute_centerlinemask(self, **kwargs):
        """Compute the centerline mask.

        Function for computing the centerline mask. The default implementation
        is a morphological skeletonization operation using the
        `skimage.morphology.skeletonize` function.

        Alternatively, the method of  centerline extraction based on non-maxima
        suppression of the singularity index, as described in [3]_ can be
        specified. This requires the optional dependency `RivaMap`_.

        .. [3] Isikdogan, Furkan, Alan Bovik, and Paola Passalacqua. "RivaMap:
               An automated river analysis and mapping engine." Remote Sensing
               of Environment 202 (2017): 88-97.

        .. _Rivamap: https://github.com/isikdogan/rivamap

        Other Parameters
        ----------------
        minScale : float, optional
            Minimum scale to use for the singularity index extraction, see [3]_

        nrScales : int, optional
            Number of scales to use for singularity index, see [3]_

        nms_threshold : float between 0 and 1, optional
            Threshold to convert the non-maxima suppression results into a
            binary mask. Default value is 0.1 which means that the lowest 10%
            non-maxima suppression values are ignored when making the binary
            centerline mask.

        """
        # skimage.morphology.skeletonize() method
        if self.method == 'skeletonize':
            for i in range(0, np.shape(self._mask)[0]):
                self._mask[i, :, :] = morphology.skeletonize(self.data[i, :, :])

        if self.method == 'rivamap':
            # rivamap based method - first check for import error
            try:
                from rivamap.singularity_index import applyMMSI as MMSI
                from rivamap.singularity_index import SingularityIndexFilters as SF
                from rivamap.delineate import extractCenterlines as eCL
            except Exception:
                raise ImportError('You must install the optional dependency:'
                                  ' rivamap, to use this centerline extraction'
                                  ' method')

            # pop the kwargs
            self.minScale = kwargs.pop('minScale', 1.5)
            self.nrScales = kwargs.pop('nrScales', 12)
            self.nms_threshold = kwargs.pop('nms_threshold', 0.1)

            # now do the computation - first change type and do psi extraction
            if self.data.dtype == 'int64':
                self.data = self.data.astype('float')/(2**64 - 1)
            self.psi, widths, orient = MMSI(self.data,
                                            filters=SF(minScale=self.minScale,
                                                       nrScales=self.nrScales))
            # compute non-maxima suppresion then normalize/threshold to
            # make binary
            self.nms = eCL(orient, self.psi)
            nms_norm = self.nms/self.nms.max()
            # compute mask
            self._mask = (nms_norm > self.nms_threshold) * 1
