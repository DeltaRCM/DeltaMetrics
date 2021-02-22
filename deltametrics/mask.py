"""Classes and methods to create masks of planform features and attributes."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from skimage import feature
from skimage import morphology
from skimage import measure

import abc

from . import utils
from . import cube

from numba import jit, njit
import xarray as xr


class BaseMask(abc.ABC):
    """Low-level base class to be inherited by all mask objects."""

    def __init__(self, mask_type, *args):
        """Initialize the base mask attributes and methods.

        This intialization tries to determine the types of intputs given to
        the initializer and proceeds from there.

        Parameters
        ----------
        mask_type : :obj:`str`
            Descriptor of the mask type.

        *args
        """
        # add mask type as attribute
        self._mask_type = mask_type

        # determine the types of inputs given
        if len(args) == 0:
            raise ValueError('Length of arguments init was 0.')
        elif (len(args) == 1) and issubclass(type(args[0]), cube.BaseCube):
            self._input_flag = 'cube'
        else:
            # check that all arguments are xarray or numpy arrays
            self._input_flag = 'array'
            for i in range(len(args)):
                if not (isinstance(args[i], xr.core.dataarray.DataArray)) or (isinstance(args[i], np.ndarray)):
                    raise TypeError('Input to mask instantiation was not an array. {}'.format(type(args[i])))

        #     elif type(data) is not np.ndarray:
        #         try:
        #             self.data = data.__array__()
        #         except Exception:
        #             raise TypeError('Input data type must be numpy.ndarray,'
        #                             'but was ' + str(type(data)))
        #     else:
        #         self.data = data

        # # set data to be 3D even if it is not (assuming t-x-y)
        # if len(np.shape(self.data)) == 3:
        #     pass
        # elif len(np.shape(self.data)) == 2:
        #     self.data = np.reshape(self.data, [1,
        #                                        np.shape(self.data)[0],
        #                                        np.shape(self.data)[1]])
        # else:
        #     raise ValueError('Input data shape was not 2-D nor 3-D')
        
        # initialize the mask to the correct size
        if self._input_flag == 'cube':
            _cube = args[0]
            self._shape = _cube.shape[1:]
            self._mask = np.zeros(self._shape)
        elif self._input_flag == 'array':
            _arr = args[0]
            self._shape = _arr.shape
            self._mask = np.zeros(self._shape)
        else:
            raise ValueError('Invalid _input_flag. Something went wrong.')

        

    @abc.abstractmethod
    def compute_mask(self):
        ...

    @property
    def mask_type(self):
        """Type of the mask (string)
        """
        return self._mask_type

    # @property
    # def data(self):
    #     """ndarray : Values of the mask object.

    #     In setter, we should sanitize the inputs (enforce range 0-1) and
    #     convert everything to uints for speed and size.
    #     """
    #     return self._data

    # @data.setter
    # def data(self, var):
    #     self._data = var

    @property
    def shape(self):
        return self._shape

    @property
    def mask(self):
        """ndarray : Binary mask values.

        Read-only mask attribute.
        """
        return self._mask

    def show(self, ax=None, **kwargs):
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
            ax.imshow(self.mask, cmap=cmap, **kwargs)
            ax.set_title('A ' + self.mask_type + ' mask')
        else:
            raise AttributeError('No mask computed and/or mask is all 0s.')
        plt.draw()

    def _check_deprecated_is_mask(self, is_mask):
        if not (is_mask is None):
            warnings.warn(DeprecationWarning('The is_mask argument is deprecated. It does not have any function now.'))


class ElevationMask(BaseMask):
    def __init__(self, *args, elevation_threshold, is_mask=None, **kwargs):
        super().__init__('elevation', *args)

        self._check_deprecated_is_mask(is_mask)

        # handle the arguments
        self._elevation_threshold = elevation_threshold
        print("ELEV THRESH:", self.elevation_threshold)

        # temporary storage of args as needed for processing
        if self._input_flag == 'cube':
            self.tval = kwargs.pop('t', -1)
            self._args = {'eta': args[0]['eta'][self.tval, :, 0]}
        elif self._input_flag == 'array':
            self._args = {'eta': args[0]}
        else:
            raise ValueError('Invalid _input_flag. Did you modify this attribute?')

        # compute the mask
        self.compute_mask(**kwargs)

        # clear the args
        self._args = None
        # use delattr() instead?

    def compute_mask(self, **kwargs):

        # trim the data
        trim_idx = utils.determine_land_width(self._args['eta'][:, 0])
        data_trim = self._args['eta'][trim_idx:, :]

        # use elevation_threshold to identify oceanmap
        omap = (data_trim > self.elevation_threshold) * 1.
        
        # set the data into the mask
        self._mask[trim_idx:, :] = omap

    @property
    def elevation_threshold(self):
        return self._elevation_threshold



    


# class ChannelMask(BaseMask, OAM):
#     """Identify a binary channel mask.

#     A channel mask object, helps enforce valid masking of channels.

#     Examples
#     --------
#     Initialize the channel mask
#         >>> velocity = rcm8cube['velocity'][-1, :, :]
#         >>> topo = rcm8cube['eta'][-1, :, :]
#         >>> cmsk = dm.mask.ChannelMask(velocity, topo)

#     And visualize the mask:
#         >>> cmsk.show()

#     .. plot:: mask/channelmask.py

#     """

#     def __init__(self,
#                  velocity,
#                  topo,
#                  velocity_threshold=0.3,
#                  angle_threshold=75,
#                  is_mask=False,
#                  **kwargs):
#         """Initialize the ChannelMask.

#         Intializing the channel mask requires a flow velocity field and an
#         array of the delta topography.

#         Parameters
#         ----------
#         velocity : ndarray
#             The velocity array to be used for mask creation.

#         topo : ndarray
#             The model topography to be used for mask creation.

#         velocity_threshold : float, optional
#             Threshold velocity above which flow is considered 'channelized'.
#             The default value is 0.3 m/s based on DeltaRCM default parameters
#             for sediment transport.

#         angle_threshold : int, optional
#             Threshold opening angle used in the OAM. Default is 75 degrees.

#         is_mask : bool, optional
#             Whether the data in :obj:`arr` is already a binary mask. Default is
#             False. This should be set to True, if you have already binarized
#             the data yourself, using custom routines, and want to just store
#             the data in the ChannelMask object.

#         Other Parameters
#         ----------------
#         landmask : :obj:`LandMask`, optional
#             A :obj:`LandMask` object with a defined binary land mask.
#             If given, it will be used to help define the channel mask.

#         wetmask : :obj:`WetMask`, optional
#             A :obj:`WetMask` object with a defined binary wet mask.
#             If given, the landmask attribute it contains will be used to
#             determine the channel mask.

#         kwargs : optional
#             Keyword arguments for :obj:`compute_shoremask`.

#         """
#         super().__init__(mask_type='channel', data=topo)
#         if type(velocity) is np.ndarray:
#             self.velocity = velocity
#         else:
#             try:
#                 self.velocity = velocity.__array__()
#             except Exception:
#                 raise TypeError('Input velocity parameter is invalid. Must be'
#                                 'a np.ndarray or a'
#                                 'deltametrics.cube.CubeVariable object but it'
#                                 'was a: ' + str(type(velocity)))

#         # assign **kwargs to self incase there are masks
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#         # write parameter values to self so they don't get lost
#         self.velocity_threshold = velocity_threshold
#         self.angle_threshold = angle_threshold

#         if is_mask is False:
#             self.compute_channelmask(**kwargs)
#         elif is_mask is True:
#             self._mask = self.data
#         else:
#             raise TypeError('is_mask must be a `bool`,'
#                             'but was: ' + str(type(is_mask)))

#     def compute_channelmask(self, **kwargs):
#         """Compute the ChannelMask.

#         Either uses an existing landmask, or recomputes it to use with the flow
#         velocity threshold to calculate a binary channelmask.

#         Other Parameters
#         ----------------
#         kwargs : optional
#             Keyword arguments for :obj:`compute_shoremask`.

#         """
#         # check if a 'landmask' is available
#         if hasattr(self, 'landmask'):
#             try:
#                 self.oceanmap = self.landmask.oceanmap
#                 self.landmask = self.landmask.mask
#             except Exception:
#                 self.compute_shoremask(self.angle_threshold, **kwargs)
#                 self.landmask = (self.shore_image < self.angle_threshold) * 1
#         elif hasattr(self, 'wetmask'):
#             try:
#                 self.landmask = self.wetmask.landmask
#             except Exception:
#                 self.compute_shoremask(self.angle_threshold, **kwargs)
#                 self.landmask = (self.shore_image < self.angle_threshold) * 1
#         else:
#             self.compute_shoremask(self.angle_threshold, **kwargs)
#             self.landmask = (self.shore_image < self.angle_threshold) * 1

#         # compute a flowmap of cells where flow exceeds the velocity threshold
#         self.flowmap = (self.velocity > self.velocity_threshold) * 1.

#         # calculate the channelmask as the cells exceeding the threshold
#         # within the topset of the delta (ignoring flow in ocean)
#         self._mask = np.minimum(self.landmask, self.flowmap)


# class WetMask(BaseMask, OAM):
#     """Compute the wet mask.

#     A wet mask object, identifies all wet pixels on the delta topset. Starts
#     with the land mask and then uses the topo_threshold defined for the
#     shoreline computation to add the wet pixels on the topset back to the mask.

#     If a land mask has already been computed, then it can be used to define the
#     wet mask. Otherwise the wet mask can be computed from scratch.

#     Examples
#     --------
#     Initialize the wet mask
#         >>> arr = rcm8cube['eta'][-1, :, :]
#         >>> wmsk = dm.mask.WetMask(arr)

#     And visualize the mask:
#         >>> wmsk.show()

#     .. plot:: mask/wetmask.py

#     """

#     def __init__(self,
#                  arr,
#                  angle_threshold=75,
#                  is_mask=False,
#                  **kwargs):
#         """Initialize the WetMask.

#         Intializing the wet mask requires either a 2-D array of data, or it
#         can be computed if a :obj:`LandMask` has been previously computed.

#         Parameters
#         ----------
#         arr : ndarray
#             The data array to make the mask from.

#         angle_threshold : int, optional
#             Threshold opening angle used in the OAM. Default is 75 degrees.

#         is_mask : bool, optional
#             Whether the data in :obj:`arr` is already a binary mask. Default is
#             False. This should be set to True, if you have already binarized
#             the data yourself, using custom routines, and want to just store
#             the data in the WetMask object.

#         Other Parameters
#         ----------------
#         landmask : :obj:`LandMask`, optional
#             A :obj:`LandMask` object with a defined binary shoreline mask.
#             If given, the :obj:`LandMask` object will be checked for the
#             `shore_image` and `angle_threshold` attributes.

#         kwargs : optional
#             Keyword arguments for :obj:`compute_shoremask`.

#         """
#         super().__init__(mask_type='wet', data=arr)

#         # assign **kwargs to self in case the landmask was passed
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#         # put angle_threshold in self
#         self.angle_threshold = angle_threshold

#         if is_mask is False:
#             self.compute_wetmask(**kwargs)
#         elif is_mask is True:
#             self._mask = self.data
#         else:
#             raise TypeError('is_mask must be a `bool`,'
#                             'but was: ' + str(type(is_mask)))

#     def compute_wetmask(self, **kwargs):
#         """Compute the WetMask.

#         Either recomputes the landmask, or uses precomputed information from
#         the landmask to create the wetmask.

#         """
#         # check if a 'landmask' was passed in
#         if hasattr(self, 'landmask'):
#             try:
#                 self.oceanmap = self.landmask.oceanmap
#                 self.landmask = self.landmask.mask
#             except Exception:
#                 self.compute_shoremask(self.angle_threshold, **kwargs)
#                 self.landmask = (self.shore_image < self.angle_threshold) * 1
#         else:
#             self.compute_shoremask(self.angle_threshold, **kwargs)
#             self.landmask = (self.shore_image < self.angle_threshold) * 1
#         # set wet mask as data.mask
#         self._mask = self.oceanmap * self.landmask


class LandMask(BaseMask):
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

    def __init__(self, *args, is_mask=None, **kwargs):
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
        super().__init__('land', *args)

        self._check_deprecated_is_mask(is_mask)

        # assign **kwargs to self in event a mask was passed
        # for key, value in kwargs.items():
        #     setattr(self, key, value)

        # set parameter value to self so it is kept
        # self.angle_threshold = angle_threshold

        if not is_mask:
            self.compute_landmask(**kwargs)
        elif is_mask is True:
            self._mask = self.data
        else:
            raise TypeError('is_mask must be a `bool`,'
                            'but was: ' + str(type(is_mask)))

    def compute_mask(self, **kwargs):
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


class ShorelineMask(BaseMask):
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
    
    @staticmethod
    def from_masks(ElevationMask):
        """Create a ShorelineMask directly from an ElevationMask.
        """
        return ShorelineMask()

    def __init__(self, *args, angle_threshold=75, t=-1, H_SL=0,
                 is_mask=None, **kwargs):
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

        Other Parameters
        ----------------
        elevation_threshold
            Passed to the initialization of an ElevationMask to discern the
            ocean area binary mask input to the opening angle method. 

        kwargs : optional
            Keyword arguments for :obj:`shaw_opening_angle_method`.

        """
        super().__init__('shoreline', *args)

        self._check_deprecated_is_mask(is_mask)

        # handle the arguments
        self._angle_threshold = angle_threshold

        # temporary storage of args as needed for processing
        if self._input_flag == 'cube':
            self.tval = kwargs.pop('t', -1)
            self._args = {'eta': args[0]['eta'][self.tval, :, 0]}
        elif self._input_flag == 'array':
            self._args = {'eta': args[0]}
        else:
            raise ValueError('Invalid _input_flag. Did you modify this attribute?')

        # initialize arrays
        self.oceanmap = np.zeros(self._shape)
        self.shore_image = np.zeros(self._shape)

        # compute the mask
        self.compute_mask(**kwargs)

        # clear the args
        self._args = None
        # use delattr() instead?

        # if is_mask is False:
        #     self.compute_shoremask(angle_threshold, **kwargs)
        # elif is_mask is True:
        #     self._mask = self.data
        # else:
        #     raise TypeError('is_mask must be a `bool`,'
        #                     'but was: ' + str(type(is_mask)))

    def compute_mask(self, **kwargs):
        """Compute the shoreline mask.

        Applies the opening angle method [1]_ to compute the shoreline mask.
        Implementation of the OAM is in :obj:`shaw_opening_angle_method`.
        
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
        _numviews = kwargs.pop('numviews', 3)
        _elevation_threshold = kwargs.pop('elevation_threshold', -0.5)



        # trim_idx = utils.guess_land_width_from_land(self.data[tval, :, 0])
        # data_trim = self.data[tval, trim_idx:, :]
        # # use topo_threshold to identify oceanmap
        # omap = (data_trim < self.topo_threshold) * 1.

        # create an elevation mask and invert it for an ocean area mask
        _em = ElevationMask(self._args['eta'], 
                            elevation_threshold=_elevation_threshold)
        omask = 1 - (_em.mask)

        # if all ocean, there is no shore to be found
        if (omap == 1).all():
            pass
        else:
            
            # apply opening angle method
            _, seaangles = shaw_opening_angle_method(omask, _numviews)

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

    @property
    def angle_threshold(self):
        """Threshold angle used for picking shoreline."""
        return self._angle_threshold
    


# class EdgeMask(BaseMask, OAM):
#     """Identify the land-water edges.

#     An edge mask object, delineates the boundaries between land and water.

#     Examples
#     --------
#     Initialize the edge mask
#         >>> arr = rcm8cube['eta'][-1, :, :]
#         >>> emsk = dm.mask.EdgeMask(arr)

#     Visualize the mask:
#         >>> emsk.show()

#     .. plot:: mask/edgemask.py

#     """

#     def __init__(self,
#                  arr,
#                  angle_threshold=75,
#                  is_mask=False,
#                  **kwargs):
#         """Initialize the EdgeMask.

#         Initializing the edge mask requires either a 2-D array of topographic
#         data, or it can be computed using the :obj:`LandMask` and the
#         :obj:`WetMask`.

#         Parameters
#         ----------
#         arr : ndarray
#             The data array to make the mask from.

#         angle_threshold : int, optional
#             Threshold opening angle used in the OAM. Default is 75 degrees.

#         is_mask : bool, optional
#             Whether the data in :obj:`arr` is already a binary mask. Default
#             value is False. This should be set to True, if you have already
#             binarized the data yourself, using custom routines, and want to
#             just store the data in the EdgeMask object.

#         Other Parameters
#         ----------------
#         landmask : :obj:`LandMask`, optional
#             A :obj:`LandMask` object with the land identified

#         wetmask : :obj:`WetMask`, optional
#             A :obj:`WetMask` object with the surface water identified

#         kwargs : optional
#             Keyword arguments for :obj:`compute_shoremask`.

#         """
#         super().__init__(mask_type='edge', data=arr)

#         # assign **kwargs to self incase there are masks
#         for key, value in kwargs.items():
#             setattr(self, key, value)

#         # write parameter to self so it is not lost
#         self.angle_threshold = angle_threshold

#         if is_mask is False:
#             self.compute_edgemask(**kwargs)
#         elif is_mask is True:
#             self._mask = self.data
#         else:
#             raise TypeError('is_mask must be a `bool`,'
#                             'but was: ' + str(type(is_mask)))

#     def compute_edgemask(self, **kwargs):
#         """Compute the EdgeMask.

#         Either computes it from just the topographic array, or uses information
#         from :obj:`LandMask` and :obj:`WetMask` to create the edge mask.

#         """
#         # if landmask and edgemask exist it is quick
#         if hasattr(self, 'landmask') and hasattr(self, 'wetmask'):
#             try:
#                 self.landmask = self.landmask.mask
#                 self.wetmask = self.wetmask.mask
#             except Exception:
#                 self.compute_shoremask(self.angle_threshold, **kwargs)
#                 self.landmask = (self.shore_image < self.angle_threshold) * 1
#                 self.wetmask = self.oceanmap * self.landmask
#         elif hasattr(self, 'wetmask'):
#             try:
#                 self.wetmask = self.wetmask.mask
#                 self.landmask = self.wetmask.landmask
#             except Exception:
#                 self.compute_shoremask(self.angle_threshold, **kwargs)
#                 self.landmask = (self.shore_image < self.angle_threshold) * 1
#                 self.wetmask = self.oceanmap * self.landmask
#         else:
#             self.compute_shoremask(self.angle_threshold, **kwargs)
#             self.landmask = (self.shore_image < self.angle_threshold) * 1
#             self.wetmask = self.oceanmap * self.landmask
#         # compute the edge mask
#         for i in range(0, np.shape(self._mask)[0]):
#             self._mask[i, :, :] = np.maximum(0,
#                                              feature.canny(self.wetmask[i,
#                                                                         :,
#                                                                         :])*1 -
#                                              feature.canny(self.landmask[i,
#                                                                          :,
#                                                                          :])*1)


# class CenterlineMask(BaseMask):
#     """Identify channel centerline mask.

#     A centerline mask object, provides the location of channel centerlines.

#     Examples
#     --------
#     Initialize the centerline mask
#         >>> channelmask = dm.mask.ChannelMask(rcm8cube['velocity'][-1, :, :],
#                                               rcm8cube['eta'][-1, :, :])
#         >>> clmsk = dm.mask.CenterlineMask(channelmask)

#     Visualize the mask
#         >>> clmsk.show()

#     .. plot:: mask/centerlinemask.py

#     """

#     def __init__(self,
#                  channelmask,
#                  is_mask=False,
#                  method='skeletonize',
#                  **kwargs):
#         """Initialize the CenterlineMask.

#         Initialization of the centerline mask object requires a 2-D channel
#         mask (can be the :obj:`ChannelMask` object or a binary 2-D array).

#         Parameters
#         ----------
#         channelmask : :obj:`ChannelMask` or ndarray
#             The channel mask to derive the centerlines from

#         is_mask : bool, optional
#             Whether the data in :obj:`arr` is already a binary mask. Default
#             value is False. This should be set to True, if you have already
#             binarized the data yourself, using custom routines, and want to
#             just store the data in the CenterlineMask object.

#         method : str, optional
#             The method to use for the centerline mask computation. The default
#             method ('skeletonize') is a morphological skeletonization of the
#             channel mask.

#         Other Parameters
#         ----------------
#         kwargs : optional
#             Keyword arguments for the 'rivamap' functionality.

#         """
#         if isinstance(channelmask, np.ndarray):
#             super().__init__(mask_type='centerline', data=channelmask)
#         elif isinstance(channelmask, ChannelMask):
#             super().__init__(mask_type='centerline',
#                              data=channelmask.mask)
#         else:
#             raise TypeError('Input channelmask parameter is invalid. Must be'
#                             'a np.ndarray or ChannelMask object but it was'
#                             'a: ' + str(type(channelmask)))

#         # save method type value to self
#         self.method = method

#         if is_mask is False:
#             self.compute_centerlinemask(**kwargs)
#         elif is_mask is True:
#             self._mask = self.data
#         else:
#             raise TypeError('is_mask must be a `bool`,'
#                             'but was: ' + str(type(is_mask)))

#     def compute_centerlinemask(self, **kwargs):
#         """Compute the centerline mask.

#         Function for computing the centerline mask. The default implementation
#         is a morphological skeletonization operation using the
#         `skimage.morphology.skeletonize` function.

#         Alternatively, the method of  centerline extraction based on non-maxima
#         suppression of the singularity index, as described in [3]_ can be
#         specified. This requires the optional dependency `RivaMap`_.

#         .. [3] Isikdogan, Furkan, Alan Bovik, and Paola Passalacqua. "RivaMap:
#                An automated river analysis and mapping engine." Remote Sensing
#                of Environment 202 (2017): 88-97.

#         .. _Rivamap: https://github.com/isikdogan/rivamap

#         Other Parameters
#         ----------------
#         minScale : float, optional
#             Minimum scale to use for the singularity index extraction, see [3]_

#         nrScales : int, optional
#             Number of scales to use for singularity index, see [3]_

#         nms_threshold : float between 0 and 1, optional
#             Threshold to convert the non-maxima suppression results into a
#             binary mask. Default value is 0.1 which means that the lowest 10%
#             non-maxima suppression values are ignored when making the binary
#             centerline mask.

#         """
#         # skimage.morphology.skeletonize() method
#         if self.method == 'skeletonize':
#             for i in range(0, np.shape(self._mask)[0]):
#                 self._mask[i, :, :] = morphology.skeletonize(self.data[i, :, :])

#         if self.method == 'rivamap':
#             # rivamap based method - first check for import error
#             try:
#                 from rivamap.singularity_index import applyMMSI as MMSI
#                 from rivamap.singularity_index import SingularityIndexFilters as SF
#                 from rivamap.delineate import extractCenterlines as eCL
#             except Exception:
#                 raise ImportError('You must install the optional dependency:'
#                                   ' rivamap, to use this centerline extraction'
#                                   ' method')

#             # pop the kwargs
#             self.minScale = kwargs.pop('minScale', 1.5)
#             self.nrScales = kwargs.pop('nrScales', 12)
#             self.nms_threshold = kwargs.pop('nms_threshold', 0.1)

#             # now do the computation - first change type and do psi extraction
#             if self.data.dtype == 'int64':
#                 self.data = self.data.astype('float')/(2**64 - 1)
#             self.psi, widths, orient = MMSI(self.data,
#                                             filters=SF(minScale=self.minScale,
#                                                        nrScales=self.nrScales))
#             # compute non-maxima suppresion then normalize/threshold to
#             # make binary
#             self.nms = eCL(orient, self.psi)
#             nms_norm = self.nms/self.nms.max()
#             # compute mask
#             self._mask = (nms_norm > self.nms_threshold) * 1


class GeometricMask(BaseMask):
    """Create simple geometric masks.

    A geometric mask object, allows for creation of subdomains in multiple
    ways. Angular boundaries can define an area of interest in the shape
    of an arc. Radial boundaries can define a radial region to mask.
    Boundaries defined in the strike direction (perpendicular to inlet
    channel), can be supplied to define a region of interest. Similarly,
    boundaries can be defined in the dip direction (parallel to inlet channel
    orientation).

    Examples
    --------
    Initialize the geometric mask using model topography as a base.
        >>> arr = rcm8cube['eta'][-1, :, :]
        >>> gmsk = dm.mask.GeometricMask(arr)

    Define an angular mask to cover half the domain from 0 to pi/2.
        >>> gmsk.angular(0, np.pi/2)

    Further mask this region by defining some bounds in the strike direction.
        >>> gmsk.strike(10, 50)

    Visualize the mask:
        >>> gmsk.show()

    .. plot:: mask/geomask.py

    """

    def __init__(self, arr, is_mask=False):
        """Initialize the GeometricMask.

        Initializing the geometric mask object requires a 2-D array of the
        region you wish to apply the mask to.

        Parameters
        ----------
        arr : ndarray
            2-D array to be masked.

        is_mask : bool, optional
            Whether the data in :obj:`arr` is already a binary mask. Default
            value is False. This should be set to True, if you have already
            binarized the data yourself, using custom routines, or want to
            further mask a pre-existing mask using geometric boundaries, this
            should be set to True.

        """
        super().__init__(mask_type='geometric', data=arr)

        if is_mask is False:
            self._mask = np.ones_like(self.data)
        elif is_mask is True:
            self._mask = self.data
        else:
            raise TypeError('is_mask must be a `bool`,'
                            'but was: ' + str(type(is_mask)))

        _, self.L, self.W = np.shape(self._mask)
        self.xc = 0
        self.yc = int(self.W/2)

    @property
    def xc(self):
        """x-coordinate of origin point."""
        return self._xc

    @xc.setter
    def xc(self, var):
        self._xc = var

    @property
    def yc(self):
        """y-coordinate of origin point."""
        return self._yc

    @yc.setter
    def yc(self, var):
        self._yc = var

    def angular(self, theta1, theta2):
        """Make a mask based on two angles.

        Computes a mask that is bounded by 2 angles input by the user.

        .. note::
           Requires a domain with a width greater than 2x its length right now.
           Function should be re-factored to be more flexible.

        .. note::
           Currently origin point is fixed, function should be extended to
           allow for an input origin point from which the angular bounds are
           determined.

        Parameters
        ----------
        theta1 : float
            Radian value controlling the left bound of the mask

        theta2 : float
            Radian value controlling the right bound of the mask

        """
        if self.L/self.W > 0.5:
            raise ValueError('Width of input array must exceed 2x length.')
        y, x = np.ogrid[0:self.W, -self.L:self.L]
        theta = np.arctan2(x, y) - theta1 + np.pi/2
        theta %= (2*np.pi)
        anglemask = theta <= (theta2-theta1)
        _, B = np.shape(anglemask)
        anglemap = anglemask[:self.L, int(B/2-self.W/2):int(B/2+self.W/2)]

        self._mask = self._mask * anglemap

    def circular(self, rad1, rad2=None, origin=None):
        """Make a circular mask bounded by two radii.

        Computes a mask that is bounded by 2 radial distances which are input
        by the user.

        Parameters
        ----------
        rad1 : int
            Index value to set the inner radius.

        rad2 : int, optional
            Index value to set the outer radius. If unspecified, this bound
            is set to extend as the larger dimension of the domain.

        origin : tuple, optional
            Tuple containing the (x, y) coordinate of the origin point.
            If unspecified, it is assumed to be at the center of a boundary
            where pyDeltaRCM places the inlet.

        """
        if origin is not None:
            self.xc = origin[0]
            self.yc = origin[1]

        if rad2 is None:
            rad2 = np.max((self.L, self.W))

        yy, xx = np.meshgrid(range(self.W), range(self.L))
        # calculate array of distances from inlet
        raddist = np.sqrt((yy-self._yc)**2 + (xx-self._xc)**2)
        # identify points within radial bounds
        raddist = np.where(raddist >= rad1, raddist, 0)
        raddist = np.where(raddist <= rad2, raddist, 0)
        raddist = np.where(raddist == 0, raddist, 1)
        # make 3D to be consistent with mask
        raddist = np.reshape(raddist, [1, self.L, self.W])
        # combine with current mask via multiplication
        self._mask = self._mask * raddist

    def strike(self, ind1, ind2=None):
        """Make a mask based on two indices.

        Makes a mask bounded by lines perpendicular to the direction of the
        flow in the inlet channel.

        Parameters
        ----------
        ind1 : int
            Index value to set first boundary (closest to inlet)

        ind2 : int, optional
            Index value to set second boundary (farther from inlet). This is
            optional, if unspecified, this is set to the length of the domain.

        """
        if ind2 is None:
            ind2 = self.L

        temp_mask = np.zeros_like(self._mask)
        temp_mask[:, ind1:ind2, :] = 1

        self._mask = self._mask * temp_mask

    def dip(self, ind1, ind2=None):
        """Make a mask parallel to the inlet.

        Makes a mask that is parallel to the direction of flow in the inlet.
        If only one value is supplied, then this mask is centered on the
        inlet and has a width of ind1. If two values are supplied then they
        set the bounds for the mask.

        Parameters
        ----------
        ind1 : int
            Width of the mask if ind2 is not specified. Otherwise, it sets
            the left bound of the mask.

        ind2 : int, optional
            Right bound of the mask if specified. If not specified, then
            ind1 sets the width of a mask centered on the inlet.

        """
        temp_mask = np.zeros_like(self._mask)
        if ind2 is None:
            w_ind = int(ind1/2)
            temp_mask[:, :, self._yc-w_ind:self._yc+w_ind+1] = 1
        else:
            temp_mask[:, :, ind1:ind2] = 1

        self._mask = self._mask * temp_mask


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


def shaw_opening_angle_method(ocean_mask, numviews=3):
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
    ocean_mask : ndarray
        Binary image that has been thresholded to split deep water/land.

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

    Sx, Sy = np.gradient(ocean_mask)
    G = np.sqrt((Sx*Sx) + (Sy*Sy))

    # threshold the gradient to produce edges
    edges = (G > 0) & (ocean_mask > 0)

    # extract coordinates of the edge pixels and define convex hull
    bordermap = np.pad(np.zeros_like(edges), 1, 'edge')
    bordermap[:-2, 1:-1] = edges
    bordermap[0, :] = 1
    points = np.fliplr(np.array(np.where(edges > 0)).T)
    hull = ConvexHull(points, qhull_options='Qc')

    # identify set of points to evaluate
    sea = np.fliplr(np.array(np.where(ocean_mask > 0.5)).T)

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
