"""Classes and methods to create masks of planform features and attributes."""
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class BaseMask(object):

    def __init__(self, mask_type, data):
        """Initialize the base mask attributes and methods.

        """
        self.mask_type = mask_type
        self.data = data

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
    def time(self):
        """float : Time when the mask is from.
        """
        return self._time

    @time.setter
    def time(self, var):
        self._time = var

    def a_method(self):
        """Does something?

        """
        pass

    def show(self, **kwargs):
        """Show the mask.

        Passes `**kwargs` to ``matplotlib.imshow``.
        """
        cmap = kwargs.pop('cmap', 'gray')
        fig, ax = plt.subplots()
        if hasattr(self, 'mask'):
            ax.imshow(self.mask, cmap=cmap, **kwargs)
            ax.set_title('A ' + self.mask_type + ' mask')
        else:
            ax.imshow(self.data, cmap=cmap, **kwargs)
            ax.set_title('No mask computed, input data shown')
        plt.show()

    def compute_shoremask(self):
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

        """
        # use the first column to trim the land_width
        # (or, can we assume access to the DeltaRCM variable `L0`?)
        i = 0
        delt = 10
        while i < self.data.shape[0] and delt != 0:
            delt = self.data[i, 0] - self.data[i+1, 0]
            i += 1
        trim_idx = i - 1  # assign the trimming index
        data_trim = self.data[trim_idx:, :]
        # use topo_threshold to identify oceanmap
        self.oceanmap = (data_trim < self.topo_threshold) * 1.
        # apply seaangles function
        _, seaangles = self.Seaangles_mod(self.numviews,
                                          self.oceanmap)
        # translate flat seaangles values to the shoreline image
        shore_image = np.zeros_like(data_trim)
        flat_inds = list(map(lambda x: np.ravel_multi_index(x,
                                                            shore_image.shape),
                             seaangles[:2, :].T.astype(int)))
        shore_image.flat[flat_inds] = seaangles[-1, :]
        # grab contour from seaangles plot corresponding to angle threshold
        cs = plt.contour(shore_image, [self.angle_threshold])
        plt.close()
        C = cs.allsegs[0][0]
        # convert this extracted contour to the shoreline mask
        shoremap = np.zeros_like(data_trim)
        flat_inds = list(map(lambda x: np.ravel_multi_index(x, shoremap.shape),
                             np.fliplr(np.round(C).astype(int))))
        shoremap.flat[flat_inds] = 1

        # write shoreline map out to data.mask
        self.mask = np.zeros(self.data.shape)
        self.mask[trim_idx:, :] = shoremap
        # assign shore_image to the mask object
        self.shore_image = shore_image

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


class ChannelMask(BaseMask):
    """Channel mask.

    A channel mask object, helps enforce valid masking of channels.

    Examples
    --------

    Initialize the channel mask
        >>> cmsk = dm.mask.ChannelMask(arr)

    And visualize the mask:
        >>> cmsk.show()

    .. plot:: mask/channelmask.py

    """

    def __init__(self, arr, **kwargs):
        """Initialize the ChannelMask.

        Intializing the channel mask requires an array of data, should be
        two-dimensional.

        Parameters
        ----------
        arr : ndarray
            The data array to make the mask from.

        routine : str
            Which routine to use to extract the mask.

        is_mask : bool, optional
            Whether the data in :obj:`arr` is already a binary mask. For
            example, this should be set to True, if you have already binarized
            the data yourself, using custom routines, and want to just store
            the data in the ChannelMask object.

        """
        super().__init__(mask_type='channel', data=arr)

        self.other_atts = 10

    @property
    def property_for_just_channels(self):
        """Who knows!
        """
        return self._property_for_just_channels

    def a_channel_function(self):
        """This is a wrapper to the below function.
        """
        return a_channel_function(self.data)


class WetMask(BaseMask):
    """Compute the wet mask.

    A wet mask object, identifies all wet pixels on the delta topset. Starts
    with the land mask and then uses the topo_threshold defined for the
    shoreline computation to add the wet pixels on the topset back to the mask.

    If a land mask has already been computed, then it can be used to define the
    wet mask. Otherwise the wet mask can be computed from scratch.

    Examples
    --------
    Initialize the wet mask
        >>> wmsk = dm.mask.WetMask(arr)

    And visualize the mask:
        >>> wmsk.show()

    .. plot:: mask/wetmask.py

    """

    def __init__(self, arr, **kwargs):
        """Initialize the WetMask.

        Intializing the wet mask requires either a 2-D array of data, or it
        can be computed if a :obj:`LandMask` has been previously computed.

        Parameters
        ----------
        arr : ndarray
            The data array to make the mask from.

        Other Parameters
        ----------------
        landmask : :obj:`LandMask`, optional
            A :obj:`LandMask` object with a defined binary shoreline mask.
            If given, the :obj:`LandMask` object will be checked for the
            `shore_image` and `angle_threshold` attributes.

        topo_threshold : float, optional
            Threshold depth to use for the OAM. Default is -0.5.

        angle_threshold : int, optional
            Threshold opening angle used in the OAM. Default is 75 degrees.

        numviews : int, optional
            Defines the number of times to 'look' for the OAM. Default is 3.

        is_mask : bool, optional
            Whether the data in :obj:`arr` is already a binary mask. For
            example, this should be set to True, if you have already binarized
            the data yourself, using custom routines, and want to just store
            the data in the LandMask object.

        """
        super().__init__(mask_type='wet', data=arr)

        # assign **kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)

        # set default values for **kwargs if they do not exist
        self.topo_threshold = getattr(self, 'topo_threshold', -0.5)
        self.angle_threshold = getattr(self, 'angle_threshold', 75)
        self.numviews = getattr(self, 'numviews', 3)
        self.is_mask = getattr(self, 'is_mask', False)

        if self.is_mask is False:
            self.compute_wetmask()
        elif self.is_mask is True:
            self.mask = self.data
        else:
            raise ValueError('is_mask must be a `bool`,'
                             'but was: ' + self.is_mask)

    def compute_wetmask(self):
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
                self.compute_shoremask()
                self.landmask = (self.shore_image < self.angle_threshold) * 1
        else:
            self.compute_shoremask()
            self.landmask = (self.shore_image < self.angle_threshold) * 1
        # set wet mask as data.mask
        self.mask = self.oceanmap * self.landmask


class LandMask(BaseMask):
    """Identify a binary mask of the delta topset.

    A land mask object, helps enforce valid masking of delta topset.

    If a shoreline mask has been computed, it can be used to help compute the
    land mask, otherwise it will be computed from scratch.

    Examples
    --------
    Initialize the mask.
        >>> lmsk = dm.mask.LandMask(arr)

    And visualize the mask:
        >>> lmsk.show()

    .. plot:: mask/landmask.py

    """

    def __init__(self, arr, **kwargs):
        """Initialize the LandMask.

        Intializing the land mask requires an array of data, should be
        two-dimensional. If a shoreline mask (:obj:`ShoreMask`) has been
        computed, then this can be used to define the land mask. Otherwise the
        necessary computations will occur from scratch.

        Parameters
        ----------
        arr : ndarray
            2-D topographic array to make the mask from.

        Other Parameters
        ----------------
        shoremask : :obj:`ShoreMask`, optional
            A :obj:`ShoreMask` object with a defined binary shoreline mask.
            If given, the :obj:`ShoreMask` object will be checked for the
            `shore_image` and `angle_threshold` attributes.

        topo_threshold : float, optional
            Threshold depth to use for the OAM. Default is -0.5.

        angle_threshold : int, optional
            Threshold opening angle used in the OAM. Default is 75 degrees.

        numviews : int, optional
            Defines the number of times to 'look' for the OAM. Default is 3.

        is_mask : bool, optional
            Whether the data in :obj:`arr` is already a binary mask. For
            example, this should be set to True, if you have already binarized
            the data yourself, using custom routines, and want to just store
            the data in the LandMask object.

        """
        super().__init__(mask_type='land', data=arr)

        # assign **kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)

        # set default values for **kwargs if they do not exist
        self.topo_threshold = getattr(self, 'topo_threshold', -0.5)
        self.angle_threshold = getattr(self, 'angle_threshold', 75)
        self.numviews = getattr(self, 'numviews', 3)
        self.is_mask = getattr(self, 'is_mask', False)

        if self.is_mask is False:
            self.compute_landmask()
        elif self.is_mask is True:
            self.mask = self.data
        else:
            raise ValueError('is_mask must be a `bool`,'
                             'but was: ' + self.is_mask)

    def compute_landmask(self):
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
                self.compute_shoremask()
        else:
            self.compute_shoremask()
        # set the land mask to data.mask
        self.mask = (self.shore_image < self.angle_threshold) * 1

    @property
    def property_for_just_land(self):
        """Who knows!
        """
        return self._property_for_just_land * 3

    def a_land_function(self):
        """This is a wrapper to the below function.
        """
        return a_land_function(self.data)


class ShoreMask(BaseMask):
    """Identify the shoreline as a binary mask.

    A shoreline mask object, provides a binary identification of shoreline
    pixels.

    Examples
    --------
    Initialize the mask.
        >>> shrmsk = dm.mask.ShoreMask(arr)

    Visualize the mask:
        >>> shrmsk.show()

    .. plot:: mask/shoremask.py

    """

    def __init__(self, arr, **kwargs):
        """Initialize the ShoreMask.

        Initializing the shoreline mask requires a 2-D array of data. The
        shoreline mask is computed using the opening angle method (OAM)
        described in [1]_. The default parameters are designed for
        DeltaRCM and follow the implementation described in [2]_.

        Parameters
        ----------
        arr : ndarray
            2-D topographic array to make the mask from.

        Other Parameters
        ----------------
        topo_threshold : float, optional
            Threshold depth to use for the OAM. Default is -0.5.

        angle_threshold : int, optional
            Threshold opening angle used in the OAM. Default is 75 degrees.

        numviews : int, optional
            Defines the number of times to 'look' for the OAM. Default is 3.

        is_mask : bool, optional
            Whether the data in :obj:`arr` is already a binary mask. For
            example, this should be set to True, if you have already binarized
            the data yourself, using custom routines, and want to just store
            the data in the ShoreMask object.

        """
        super().__init__(mask_type='shore', data=arr)

        # assign **kwargs to self
        for key, value in kwargs.items():
            setattr(self, key, value)

        # set default values for **kwargs if they do not exist
        self.topo_threshold = getattr(self, 'topo_threshold', -0.5)
        self.angle_threshold = getattr(self, 'angle_threshold', 75)
        self.numviews = getattr(self, 'numviews', 3)
        self.is_mask = getattr(self, 'is_mask', False)

        if self.is_mask is False:
            self.compute_shoremask()
        elif self.is_mask is True:
            self.mask = self.data
        else:
            raise ValueError('is_mask must be a `bool`,'
                             'but was: ' + self.is_mask)
