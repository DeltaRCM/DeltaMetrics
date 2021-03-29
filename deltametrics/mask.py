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
import warnings

from . import utils
from . import cube
from . import plan

from numba import jit, njit
import xarray as xr


class BaseMask(abc.ABC):
    """Low-level base class to be inherited by all mask objects."""

    def __init__(self, mask_type, *args, **kwargs):
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
        self._shape = None
        self._mask = None

        # pop is_mask, check if any value was supplied
        is_mask = kwargs.pop('is_mask', None)
        self._check_deprecated_is_mask(is_mask)

        # determine the types of inputs given
        if len(args) == 0:
            self._input_flag = None
            _allow_empty = kwargs.pop('allow_empty', False)
            if _allow_empty:
                # do nothing and return partially instantiated object
                return
            else:
                raise ValueError('Expected 1 input, got 0.')
        elif (len(args) == 1) and issubclass(type(args[0]), cube.BaseCube):
            self._input_flag = 'cube'
            self._set_shape_mask(args[0].shape[1:])
        elif (len(args) >= 1) and issubclass(type(args[0]), BaseMask):
            self._input_flag = 'mask'
            self._set_shape_mask(args[0].shape)
        elif utils.is_ndarray_or_xarray(args[0]):
            # check that all arguments are xarray or numpy arrays
            self._input_flag = 'array'
            for i in range(len(args)):
                if not utils.is_ndarray_or_xarray(args[i]):
                    raise TypeError(
                        'Input to mask instantiation was '
                        'not an array. Input type was {}'.format(
                            type(args[i])))
            self._set_shape_mask(args[0].shape)
        else:
            raise TypeError(
                'Unexpected type was input: {0}'.format(type(args[0])))

        self._check_deprecated_3d_input(args[0].shape)

    def _set_shape_mask(self, _shape):
        """Set the mask shape.

        This is called during instantiation in most standard use-cases and
        pathways, however, if you are implementing your own mask or your own
        static method to create a mask, you may need to manually call this
        method to set the fields of the mask correctly before doing any mask
        computation.

        Parameters
        ----------
        shape : :obj:`tuple`
            Shape of the mask.
        """
        self._shape = _shape
        self._mask = np.zeros(self._shape, dtype=np.bool)

    @abc.abstractmethod
    def _compute_mask(self):
        ...

    @property
    def mask_type(self):
        """Type of the mask (string)
        """
        return self._mask_type

    @property
    def shape(self):
        return self._shape

    @property
    def mask(self):
        """ndarray : Binary mask values.

        .. important::

            `mask` is a boolean array (**not integer**). See also
            :obj:`integer_mask`.

        Read-only mask attribute.
        """
        return self._mask

    @property
    def integer_mask(self):
        """ndarray : Binary mask values as integer

        .. important::

            `integer_mask` is a boolean array as ``0`` and ``1`` (integers).
            It is **not suitible** for multidimensional array indexing; see
            also :obj:`mask`.

        Read-only mask attribute.
        """
        return self._mask.astype(np.int)

    def show(self, ax=None, **kwargs):
        """Show the mask.

        The `Mask` is shown in a `matplotlib` axis with `imshow`. The `mask`
        values are accessed from :obj:`integer_mask`, so the display will show
        as ``0`` for ``False`` and ``1`` for ``True``. Default colormap is
        black and white.

        .. hint::

            Passes `**kwargs` to ``matplotlib.imshow``.

        Parameters
        ----------
        ax : :obj:`matplotlib.pyplot.Axes`
            Which axes object to plot into.

        """
        if not ax:
            ax = plt.gca()
        cmap = kwargs.pop('cmap', 'gray')
        if self._mask is None:
            raise RuntimeError(
                '`mask` field has not been intialized yet. '
                'If this is unexpected, please report error.')

        ax.imshow(self.integer_mask, cmap=cmap, **kwargs)

        plt.draw()

    def _check_deprecated_is_mask(self, is_mask):
        if not (is_mask is None):
            warnings.warn(DeprecationWarning(
                'The `is_mask` argument is deprecated. '
                'It does not have any functionality.'))

    def _check_deprecated_3d_input(self, args_0_shape):
        if self._input_flag == 'array':
            if len(args_0_shape) > 2:
                raise ValueError(
                    'Creating a `Mask` with a time dimension is deprecated. '
                    'Please manage multiple masks manually (e.g., '
                    'append the masks into a `list`).')


class ThresholdValueMask(BaseMask, abc.ABC):
    """Threshold value mask.

    This mask implements a binarization of a raster based on a threshold
    values.
    """
    def __init__(self, *args, threshold, cube_key=None, **kwargs):

        self._threshold = threshold

        # temporary storage of args as needed for processing
        if self._input_flag is None:
            # do nothing, will need to call ._compute_mask later
            return
        elif self._input_flag == 'cube':
            _tval = kwargs.pop('t', -1)
            self._field = args[0][cube_key][_tval, :, :]
        elif self._input_flag == 'array':
            self._field = args[0]
        else:
            raise ValueError(
                'Invalid _input_flag. Did you modify this attribute?')

    def _compute_threshold(self, _field):

        # use _threshold to identify field
        threshed = (_field > self._threshold) * 1.

        # set the data into the mask
        return threshed

    @property
    def field(self):
        return self._field


class ElevationMask(ThresholdValueMask):
    """Elevation mask.

    This mask implements a binarization of a raster based on elevation
    values.
    """
    def __init__(self, *args, elevation_threshold, **kwargs):

        BaseMask.__init__(self, 'elevation', *args, **kwargs)
        ThresholdValueMask.__init__(self, *args, threshold=elevation_threshold,
                                    cube_key='eta')

        # define the *args arguments needed to process this mask
        _eta = self._field

        self._compute_mask(_eta, **kwargs)

    def _compute_mask(self, _eta, **kwargs):

        # trim the data
        trim_idx = utils.determine_land_width(_eta[:, 0])
        data_trim = _eta[trim_idx:, :]

        # use elevation_threshold to identify field
        emap = self._compute_threshold(data_trim)

        # set the data into the mask
        self._mask[trim_idx:, :] = emap

    @property
    def elevation_threshold(self):
        return self._threshold

    @property
    def elevation(self):
        return self._field


class FlowMask(ThresholdValueMask):
    """Flow field mask.

    This mask implements a binarization of a raster based on flow field
    values, and can work for velocity, depth, discharge, etc.

    If you pass a cube, we will try the 'velocity' field. To use a different
    field from the cube, specify the :obj:`cube_key` argument.
    """
    def __init__(self, *args, flow_threshold,
                 cube_key='velocity', **kwargs):

        BaseMask.__init__(self, 'velocity', *args, **kwargs)
        ThresholdValueMask.__init__(self, *args, threshold=flow_threshold,
                                    cube_key=cube_key)

        # define the *args arguments needed to process this mask
        _flow = self._field

        self._compute_mask(_flow, **kwargs)

    def _compute_mask(self, _flow, **kwargs):

        # use flow_threshold to identify field
        fmap = self._compute_threshold(_flow)

        # set the data into the mask
        self._mask = fmap

    @property
    def flow_threshold(self):
        return self._threshold

    @property
    def flow(self):
        return self._field


class ChannelMask(BaseMask):
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

    @staticmethod
    def from_OAP_and_FlowMask(_OAP, _FlowMask, **kwargs):
        """Create from an OAP and a FlowMask.
        """
        # set up the empty shoreline mask
        _CM = ChannelMask(allow_empty=True, **kwargs)
        _CM._set_shape_mask(_OAP.shape)

        # set up the needed flow mask and landmask
        _LM = LandMask.from_OAP(_OAP, **kwargs)
        _FM = _FlowMask

        # compute the mask

        _CM._compute_mask(_LM, _FM, **kwargs)
        return _CM

    @staticmethod
    def from_OAP(*args, **kwargs):
        # undocumented, hopefully helpful error
        #   Note, an alternative here is to implement this method and take an
        #   OAP and information to create a flow mask, raising an error if the
        #   flow information is missing.
        raise NotImplementedError(
            '`from_OAP` is not defined for `ChannelMask` instantiation '
            'because the process additionally requires flow field '
            'information. Consider alternative methods '
            '`from_OAP_and_FlowMask()')

    @staticmethod
    def from_masks(*args, **kwargs):
        # undocumented, for convenience
        return ChannelMask.from_mask(*args, **kwargs)

    @staticmethod
    def from_mask(*args, **kwargs):
        """Create a `ChannelMask` directly from another mask.

        Can take either an ElevationMask or LandMask, and a
        FlowMask as input.

        Examples
        --------

        """
        if len(args) == 2:
            for UnknownMask in args:
                if isinstance(UnknownMask, ElevationMask):
                    # make intermediate shoreline mask
                    _LM = LandMask.from_mask(UnknownMask, **kwargs)
                elif isinstance(UnknownMask, LandMask):
                    _LM = UnknownMask
                elif isinstance(UnknownMask, FlowMask):
                    _FM = UnknownMask
                else:
                    raise TypeError('type was %s' % type(UnknownMask))
        else:
            raise ValueError(
                'Must pass two Masks to static `from_mask` for ChannelMask')

        # set up the empty shoreline mask
        _CM = ChannelMask(allow_empty=True)
        _CM._set_shape_mask(_LM.shape)

        # compute the mask
        _CM._compute_mask(_LM, _FM, **kwargs)
        return _CM

    @staticmethod
    def from_array(_arr):
        """Create a ChannelMask from an array.

        .. note::

            Instantiation with `from_array` will attempt to any data type
            (`dtype`) to boolean. This may have unexpected results. Convert
            your array to a boolean before using `from_array` to ensure the
            mask is created correctly.

        Parameters
        ----------
        _arr : :obj:`ndarray`
            The array with values to set as the mask. Can be any `dtype` but
            will be coerced to `boolean`.
        """
        # set directly
        raise NotImplementedError

    def __init__(self, *args, is_mask=None, **kwargs):
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
        super().__init__('channel', *args, **kwargs)

        # temporary storage of args as needed for processing
        if self._input_flag is None:
            # do nothing, will need to call ._compute_mask later
            return

        elif self._input_flag == 'cube':
            raise NotImplementedError
            _tval = kwargs.pop('t', -1)
            _eta = args[0]['eta'][_tval, :, :]
            _flow = args[0]['velocity'][_tval, :, :]
            # need to convert these fields to proper masks

        elif self._input_flag == 'mask':
            # this pathway should allow someone to specify a combination of
            # elevation mask, landmask, and velocity mask to make the new mask.
            raise NotImplementedError

        elif self._input_flag == 'array':
            # first make a landmas
            _eta = args[0]
            _lm = LandMask(_eta, **kwargs)
            _flow = args[1]
            _fm = FlowMask(_flow, **kwargs)

        else:
            raise ValueError('Invalid _input_flag. Did you modify this attribute?')

        # process to make the mask
        self._compute_mask(_lm, _fm, **kwargs)

    def _compute_mask(self, *args, **kwargs):
        """Compute the ChannelMask.

        Note that this method in implementation should rely only on *array*
        masks, not on the dm.mask objects.

        For this Mask, we require a landmask array and a flowmask array.
        """
        if len(args) != 2:
            raise ValueError

        if isinstance(args[0], LandMask):
            lm_array = args[0]._mask
            fm_array = args[1]._mask
        elif utils.is_ndarray_or_xarray(args[0]):
            lm_array = args[0]
            fm_array = args[1]
        else:
            raise TypeError

        # calculate the channelmask as the cells exceeding the threshold
        #   within the topset of the delta (ignoring flow in ocean)
        self._mask = np.logical_and(lm_array, fm_array)


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
        >>> arr = rcm8cube['eta'][-1, :, :]
        >>> wmsk = dm.mask.WetMask(arr)

    And visualize the mask:
        >>> wmsk.show()

    .. plot:: mask/wetmask.py

    """

    # @staticmethod
    # def from_OAP_and_ElevationMask(_OAP, _ElevationMask, **kwargs):

    #     # set up the empty shoreline mask
    #     _CM = WetMask(allow_empty=True, **kwargs)
    #     _CM._set_shape_mask(_OAP.shape)

    #     # set up the needed flow mask and landmask
    #     _LM = LandMask.from_OAP(_OAP)
    #     _EM = _ElevationMask

    #     # compute the mask
    #     _CM._compute_mask(_LM, _EM)
    #     return _CM

    @staticmethod
    def from_OAP(_OAP, **kwargs):
        """Create from an OAP.
        """
        # set up the empty shoreline mask
        _CM = WetMask(allow_empty=True, **kwargs)
        _CM._set_shape_mask(_OAP.shape)

        # set up the needed flow mask and landmask
        _LM = LandMask.from_OAP(_OAP)
        _below_mask = _OAP.below_mask

        # compute the mask (pass as arrays!)
        _CM._compute_mask(_LM._mask, _below_mask)
        return _CM

    @staticmethod
    def from_masks(*args, **kwargs):
        # undocumented, for convenience
        return WetMask.from_mask(*args, **kwargs)

    @staticmethod
    def from_mask(*args, **kwargs):
        """Create a WetMask directly from another mask.

        Needs both an ElevationMask and a LandMask, or just an ElevationMask
        and will make a LandMask internally (creates a
        `~dm.plan.OpeningAnglePlanform`); consider alternative static method
        :obj:`from_OAP_and_ElevationMask` if you are computing many masks.

        Examples
        --------

        """
        if len(args) == 2:
            # one must be ElevationMask and one LandMask
            for UnknownMask in args:
                if isinstance(UnknownMask, ElevationMask):
                    _EM = UnknownMask
                elif isinstance(UnknownMask, LandMask):
                    _LM = UnknownMask
                else:
                    raise TypeError(
                        'Double `Mask` input types must be `ElevationMask` '
                        'and `LandMask`, but received argument of type '
                        '`{0}`.'.format(type(UnknownMask)))
        elif len(args) == 1:
            UnknownMask = args[0]
            # must be ElevationMask, will create LandMask
            if isinstance(UnknownMask, ElevationMask):
                _EM = UnknownMask
                _LM = LandMask.from_mask(UnknownMask)
            else:
                raise TypeError(
                    'Single `Mask` input was expected to be type '
                    '`ElevationMask`, but was `{0}`'.format(
                        type(UnknownMask)))
        else:
            raise ValueError(
                'Must pass either one or two Masks to static method '
                '`from_mask` for `WetMask`, but received {0} args'.format(
                    len(args)))

        # set up the empty shoreline mask
        _WM = WetMask(allow_empty=True)
        _WM._set_shape_mask(_LM.shape)

        # compute the mask
        _WM._compute_mask(_LM, _EM, **kwargs)
        return _WM

    @staticmethod
    def from_array(_arr):
        """Create a WetMask from an array.

        .. note::

            Instantiation with `from_array` will attempt to any data type
            (`dtype`) to boolean. This may have unexpected results. Convert
            your array to a boolean before using `from_array` to ensure the
            mask is created correctly.

        Parameters
        ----------
        _arr : :obj:`ndarray`
            The array with values to set as the mask. Can be any `dtype` but
            will be coerced to `boolean`.
        """
        # set directly
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
        """Initialize the WetMask.

        Intializing the wet mask requires either a 2-D array of data, or it
        can be computed if a :obj:`LandMask` has been previously computed.

        .. hint::

            Pass keyword arguments to control the behavior of creating
            intermediate `Mask` or `OpeningAnglePlanform` objects.

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
            `sea_angles` and `angle_threshold` attributes.

        kwargs : optional
            Keyword arguments are passed to `LandMask` and `ElevationMask`, as
            appropriate.

        """
        super().__init__('wet', *args, **kwargs)

        # temporary storage of args as needed for processing
        if self._input_flag is None:
            # do nothing, will need to call ._compute_mask later
            return
        elif self._input_flag == 'cube':
            raise NotImplementedError
            # _tval = kwargs.pop('t', -1)
            # _eta = args[0]['eta'][_tval, :, :]
        elif self._input_flag == 'mask':
            # this pathway should allow someone to specify a combination of
            #    landmask, and ocean/elevation mask
            raise NotImplementedError
        elif self._input_flag == 'array':
            _eta = args[0]
            # first make a landmask
            _lm = LandMask(_eta, **kwargs)._mask
            # requires elevation_threshold to be in kwargs
            if 'elevation_threshold' in kwargs:
                _em = ElevationMask(_eta, **kwargs)
            else:
                raise ValueError(
                    'You must supply the keyword argument '
                    '`elevation_threshold` if instantiating a `WetMask` '
                    'directly from arrays (it is used to create an '
                    '`ElevationMask` internally).')
            # pull the wet area as the area below the elevation threshold
            _below_mask = ~_em._mask
        else:
            raise ValueError(
                'Invalid _input_flag. Did you modify this attribute?')

        # process to make the mask
        self._compute_mask(_lm, _below_mask, **kwargs)

    def _compute_mask(self, *args, **kwargs):
        """Compute the WetMask.

        Requires arrays of area below the "sea level" and landmask. There are
        multiple ways these can be supplied. For example, a LandMask object
        and an ElevationMask object, or two arrays *already corrected* for
        ocean pixels.
        """
        if len(args) == 2:
            if isinstance(args[0], LandMask):
                lm_array = args[0]._mask
                below_array = 1 - args[1]._mask  # go from elevation mask
            elif utils.is_ndarray_or_xarray(args[0]):
                lm_array = args[0]
                below_array = args[1]
            else:
                raise TypeError(
                    'Type must be array but was %s' % type(args[0]))
        else:
            raise ValueError

        # calculate the WetMask
        self._mask = np.logical_and(below_array, lm_array)


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

    @staticmethod
    def from_OAP(_OAP, **kwargs):
        # set up the empty shoreline mask
        _LM = LandMask(allow_empty=True, **kwargs)
        _LM._set_shape_mask(_OAP.shape)

        # compute the mask
        _LM._compute_mask(_OAP)
        return _LM

    @staticmethod
    def from_masks(UnknownMask):
        # undocumented, for convenience
        return LandMask.from_mask(UnknownMask)

    @staticmethod
    def from_mask(UnknownMask, **kwargs):
        """Create a ShorelineMask directly from another mask.

        Can take either an ElevationMask as input.
        """
        if isinstance(UnknownMask, ElevationMask):
            # make intermediate shoreline mask
            _OAP = plan.OpeningAnglePlanform.from_mask(
                UnknownMask, **kwargs)
        else:
            raise TypeError

        # set up the empty shoreline mask
        _LM = LandMask(allow_empty=True)
        _LM._set_shape_mask(_OAP.shape)

        # compute the mask
        _sea_angles = _OAP.sea_angles
        _LM._compute_mask(_sea_angles, **kwargs)
        return _LM

    @staticmethod
    def from_array(_arr):
        """Create a LandMask from an array.

        .. note::

            Instantiation with `from_array` will attempt to any data type
            (`dtype`) to boolean. This may have unexpected results. Convert
            your array to a boolean before using `from_array` to ensure the
            mask is created correctly.

        Parameters
        ----------
        _arr : :obj:`ndarray`
            The array with values to set as the mask. Can be any `dtype` but
            will be coerced to `boolean`.
        """
        # set directly
        raise NotImplementedError

    def __init__(self, *args, angle_threshold=75, **kwargs):
        """Initialize the LandMask.

        Intializing the land mask requires an array of data, should be
        two-dimensional.

        .. note::

            This class currently computes the mask via the Shaw opening
            angle method (:obj:`~dm.plan.shaw_opening_angle_method`). However,
            it could/should be generalized to support multiple implementations
            via a `method` argument. Then, the `angle_threshold` might not be
            a property any longer, and should be treated just as any keyword
            passed to the method for instantiation.

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
            `sea_angles` and `angle_threshold` attributes.

        kwargs : optional
            Keyword arguments for :obj:`compute_shoremask`.

        """
        super().__init__('land', *args, **kwargs)

        self._angle_threshold = angle_threshold

        # temporary storage of args as needed for processing
        if self._input_flag is None:
            # do nothing, will need to call ._compute_mask later
            return

        elif self._input_flag == 'cube':
            raise NotImplementedError
            # _tval = kwargs.pop('t', -1)
            # _eta = args[0]['eta'][_tval, :, :]

        elif self._input_flag == 'mask':
            raise NotImplementedError

        elif self._input_flag == 'array':
            _eta = args[0]

        else:
            raise ValueError(
                'Invalid _input_flag. Did you modify this attribute?')

        # create an OAP to determine the sea_angles
        _OAP = plan.OpeningAnglePlanform.from_elevation_data(
            _eta,
            **kwargs)

        # get fields out of the OAP
        _sea_angles = _OAP._sea_angles

        # make the mask, all we need is the sea angles.
        #   See not above about how this method could be better generalized.
        self._compute_mask(_sea_angles, **kwargs)

    def _compute_mask(self, *args, **kwargs):
        """Compute the LandMask.

        This method (as implemented, see note in __init__) requires the
        `sea_angles` field from the Shaw opening angle method.  This
        information can come from multiple data sources though.

        Thus, the argument to this method should be one of:
            * an opening angle planform object
            * an ndarray with the sea_angles data directly (matching shape of
              mask)

        """
        # for landmask, we need the shore image field of the OAP
        if len(args) == 1:
            if isinstance(args[0], plan.OpeningAnglePlanform):
                sea_angles = args[0]._sea_angles
            elif utils.is_ndarray_or_xarray(args[0]):
                sea_angles = args[0]
            else:
                raise TypeError
        else:
            raise ValueError('Specify only 1 argument.')

        if np.all(sea_angles == 0):
            self._mask = np.zeros(self._shape, dtype=np.bool)
        else:
            self._mask = (sea_angles < self._angle_threshold)

    @property
    def angle_threshold(self):
        """Threshold view angle used for picking land area."""
        return self._angle_threshold


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
    def from_OAP(_OAP, **kwargs):
        # set up the empty shoreline mask
        _SM = ShorelineMask(allow_empty=True, **kwargs)
        _SM._set_shape_mask(_OAP.shape)

        # compute the mask
        _SM._compute_mask(_OAP)
        return _SM

    @staticmethod
    def from_mask(UnknownMask, **kwargs):
        """Create a ShorelineMask directly from an ElevationMask.
        """
        if not isinstance(UnknownMask, ElevationMask):
            # make intermediate shoreline mask
            raise TypeError('Input must be ElevationMask')

        _OAP = plan.OpeningAnglePlanform.from_ElevationMask(UnknownMask)
        return ShorelineMask.from_OAP(_OAP, **kwargs)

    @staticmethod
    def from_masks(UnknownMask, **kwargs):
        # undocumented but for convenience
        return ShorelineMask.from_mask(UnknownMask, **kwargs)

    @staticmethod
    def from_array(_arr):
        """Create a ShorelineMask from an array.

        .. note::

            Instantiation with `from_array` will attempt to any data type
            (`dtype`) to boolean. This may have unexpected results. Convert
            your array to a boolean before using `from_array` to ensure the
            mask is created correctly.

        Parameters
        ----------
        _arr : :obj:`ndarray`
            The array with values to set as the mask. Can be any `dtype` but
            will be coerced to `boolean`.
        """
        _SM = ShorelineMask(allow_empty=True)
        _SM._angle_threshold = None
        _SM._input_flag = None
        _SM._mask = _arr.astype(np.bool)  # set the array as mask
        return _SM

    def __init__(self, *args, angle_threshold=75, **kwargs):
        """Initialize the ShorelineMask.

        Initializing the shoreline mask requires a 2-D array of data. The
        shoreline mask is computed using the opening angle method (OAM)
        described in [1]_. The default parameters are designed for
        DeltaRCM and follow the implementation described in [2]_.

        .. note::

            This class currently computes the mask via the Shaw opening
            angle method (:obj:`~dm.plan.shaw_opening_angle_method`). However,
            it could/should be generalized to support multiple implementations
            via a `method` argument. For example, a sobel edge detection and
            morpholigcal thinning on a LandMask (already made from the OAM, or
            not) may also return a good approximation of the shoreline.

        Parameters
        ----------
        arr : ndarray
            2-D topographic array to make the mask from.

        angle_threshold : float, optional
            Threshold opening angle used to determine shoreline contour based
            on the sea_angles from the OpeningAngleMethod.

        Other Parameters
        ----------------
        elevation_threshold
            Passed to the initialization of an ElevationMask to discern the
            ocean area binary mask input to the opening angle method.

        kwargs : optional
            Keyword arguments for
            :obj:`~deltametrics.plan.shaw_opening_angle_method`.

        """
        super().__init__('shoreline', *args, **kwargs)

        # begin processing the arguments and making the mask
        self._angle_threshold = angle_threshold

        # temporary storage of args as needed for processing
        if self._input_flag is None:
            # do nothing, will need to call
            #    self._compute_mask() directly later
            return

        elif self._input_flag == 'cube':
            raise NotImplementedError
            # _tval = kwargs.pop('t', -1)
            # _eta = args[0]['eta'][_tval, :, 0]

        elif self._input_flag == 'array':
            # input assumed to be array, with *elevation*
            _eta = args[0]

        elif self._input_flag == 'mask':
            raise NotImplementedError
            # must be type ElevationMask
            if not isinstance(args[0], ElevationMask):
                raise TypeError('Input `mask` must be type ElevationMask.')

        else:
            raise ValueError(
                'Invalid _input_flag. Did you modify this attribute?')

        # use an OAP to get the ocean mask and sea angles fields
        _OAP = plan.OpeningAnglePlanform.from_elevation_data(
            _eta,
            **kwargs)

        # get fields out of the OAP
        _below_mask = _OAP._below_mask
        _sea_angles = _OAP._sea_angles

        # compute the mask
        self._compute_mask(_below_mask, _sea_angles)

    def _compute_mask(self, *args, **kwargs):
        """Compute the shoreline mask.

        Applies the opening angle method [1]_ to compute the shoreline mask.
        Implementation of the OAM is in
        :obj:`~deltametrics.plan.shaw_opening_angle_method`.

        Parameters
        ----------

        Other Parameters
        ----------------
        topo_threshold : float, optional
            Threshold depth to use for the OAM. Default is -0.5.

        numviews : int, optional
            Defines the number of times to 'look' for the OAM. Default is 3.

        """
        if len(args) == 1:
            if not isinstance(args[0], plan.OpeningAnglePlanform):
                raise TypeError('Must be type OAP.')
            _below_mask = args[0]._below_mask
            _sea_angles = args[0]._sea_angles
        elif len(args) == 2:
            _below_mask = args[0]
            _sea_angles = args[1]
        else:
            raise ValueError

        # preallocate
        shoremap = np.zeros(self._shape)

        # if all ocean, there is no shore to be found
        if (_below_mask == 1).all():
            pass
        else:
            # grab contour from sea_angles corresponding to angle threshold
            cs = measure.find_contours(_sea_angles,
                                       self.angle_threshold)
            C = cs[0]

            # convert this extracted contour to the shoreline mask
            flat_inds = list(map(
                lambda x: np.ravel_multi_index(x, shoremap.shape),
                np.round(C).astype(int)))
            shoremap.flat[flat_inds] = 1

        # write shoreline map out to data.mask
        self._mask = np.copy(shoremap.astype(np.bool))

        # assign sea_angles to the mask object with proper size
        # self._sea_angles = sea_angles

        # properly assign the oceanmap to the self.oceanmap
        # self._oceanmap = _below_mask

    @property
    def angle_threshold(self):
        """Threshold angle used for picking shoreline contour.
        """
        return self._angle_threshold


class EdgeMask(BaseMask):
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

    @staticmethod
    def from_OAP_and_WetMask(_OAP, _WetMask, **kwargs):
        """Create from an OAP and a WetMask.

        .. important::

            This instantiation pathway should only be used for custom WetMask
            implementations, where the :obj:`below_mask` from the
            :obj:`OpeningAnglePlanform` is not sufficient to capture the wet
            area of the delta.

        .. hint::

            To create an EdgeMask from an existing `WetMask` *and* `LandMask`,
            see the static method :obj:`from_mask`.

        Examples
        --------
        """
        # set up the empty edge mask mask
        _EGM = EdgeMask(allow_empty=True, **kwargs)
        _EGM._set_shape_mask(_OAP.shape)

        # set up the needed flow mask and landmask
        _LM = LandMask.from_OAP(_OAP, **kwargs)
        if isinstance(_WetMask, WetMask):
            _WM = _WetMask
        else:
            raise TypeError

        # compute the mask
        _EGM._compute_mask(_LM, _WM)
        return _EGM

    @staticmethod
    def from_OAP(_OAP, **kwargs):
        """Create EdgeMask from an OAP.
        """
        _EGM = EdgeMask(allow_empty=True, **kwargs)
        _EGM._set_shape_mask(_OAP.shape)

        # set up the needed flow mask and landmask
        _LM = LandMask.from_OAP(_OAP, **kwargs)
        _WM = WetMask.from_OAP(_OAP, **kwargs)

        # compute the mask
        _EGM._compute_mask(_LM, _WM)
        return _EGM

    @staticmethod
    def from_mask(*args, **kwargs):
        """Create a EdgeMask directly from a LandMask and a WetMask.
        """
        if len(args) == 2:
            # one must be ElevationMask and one LandMask
            for UnknownMask in args:
                if isinstance(UnknownMask, WetMask):
                    _WM = UnknownMask
                elif isinstance(UnknownMask, LandMask):
                    _LM = UnknownMask
                else:
                    raise TypeError(
                        'Double `Mask` input types must be `WetMask` '
                        'and `LandMask`, but received argument of type '
                        '`{0}`.'.format(type(UnknownMask)))
        else:
            raise ValueError(
                'Must pass either one or two Masks to static method '
                '`from_mask` for `WetMask`, but received {0} args'.format(
                    len(args)))

        # set up the empty shoreline mask
        _EGM = EdgeMask(allow_empty=True)
        _EGM._set_shape_mask(_LM.shape)

        # compute the mask
        _EGM._compute_mask(_LM, _WM, **kwargs)
        return _EGM

    @staticmethod
    def from_masks(*args, **kwargs):
        # undocumented but for convenience
        return EdgeMask.from_mask(*args, **kwargs)

    @staticmethod
    def from_array(_arr):
        """Create an EdgeMask from an array.

        .. note::

            Instantiation with `from_array` will attempt to any data type
            (`dtype`) to boolean. This may have unexpected results. Convert
            your array to a boolean before using `from_array` to ensure the
            mask is created correctly.

        Parameters
        ----------
        _arr : :obj:`ndarray`
            The array with values to set as the mask. Can be any `dtype` but
            will be coerced to `boolean`.
        """
        raise NotImplementedError

    def __init__(self, *args, **kwargs):
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
        super().__init__('edge', *args, **kwargs)

        # temporary storage of args as needed for processing
        if self._input_flag is None:
            # do nothing, will need to call
            #    self._compute_mask() directly later
            return

        elif self._input_flag == 'cube':
            raise NotImplementedError
            # _tval = kwargs.pop('t', -1)
            # _eta = args[0]['eta'][_tval, :, 0]

        elif self._input_flag == 'array':
            # input assumed to be array
            _eta = args[0]

        elif self._input_flag == 'mask':
            # must be one of LandMask and WetMask
            raise NotImplementedError()

        else:
            raise ValueError(
                'Invalid _input_flag. Did you modify this attribute?')

        # make the required Masks. Both need OAP
        _OAP = plan.OpeningAnglePlanform.from_elevation_data(
                _eta,
                **kwargs)

        # get Masks from the OAP
        _LM = LandMask.from_OAP(_OAP, **kwargs)
        _WM = WetMask.from_OAP(_OAP, **kwargs)

        # compute the mask
        self._compute_mask(_LM, _WM)

    def _compute_mask(self, *args, **kwargs):
        """Compute the EdgeMask.

        This method requires arrays of LandMask and EdgeMask. These can be
        passed as either Mask objects, or as arrays.

        """
        if len(args) == 2:
            if isinstance(args[0], LandMask):
                lm_array = args[0]._mask.astype(np.float)
                wm_array = args[1]._mask.astype(np.float)
            elif utils.is_ndarray_or_xarray(args[0]):
                lm_array = args[0].astype(np.float)
                wm_array = args[1].astype(np.float)
            else:
                raise TypeError(
                    'Type must be array but was %s' % type(args[0]))
        else:
            raise ValueError(
                'Must supply `LandMask` and `WetMask` information.')

        # compute the mask with canny edge detection
        #   the arrays must be type float for this to work!
        self._mask = np.maximum(
                0, (feature.canny(wm_array) * 1 -
                    feature.canny(lm_array) * 1)).astype(np.bool)


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
    @staticmethod
    def from_OAP_and_FlowMask(_OAP, _FlowMask, **kwargs):
        """Create from an OAP and a FlowMask.
        """
        # set up the empty shoreline mask
        _CntM = CenterlineMask(allow_empty=True, **kwargs)
        _CntM._set_shape_mask(_OAP.shape)

        # set up the needed flow mask and channelmask
        _FM = _FlowMask
        _CM = ChannelMask.from_OAP_and_FlowMask(_OAP, _FM, **kwargs)

        # compute the mask

        _CntM._compute_mask(_CM, **kwargs)
        return _CntM

    @staticmethod
    def from_OAP(*args, **kwargs):
        # undocumented, hopefully helpful error
        #   Note, an alternative here is to implement this method and take an
        #   OAP and information to create a flow mask, raising an error if the
        #   flow information is missing.
        raise NotImplementedError(
            '`from_OAP` is not defined for `CenterlineMask` instantiation '
            'because the process additionally requires flow field '
            'information. Consider alternative methods '
            '`from_OAP_and_FlowMask()')

    @staticmethod
    def from_masks(*args, **kwargs):
        # undocumented, for convenience
        return CenterlineMask.from_mask(*args, **kwargs)

    @staticmethod
    def from_mask(*args, **kwargs):
        """Create a `CenterlineMask` directly from another mask.

        Can take either an ElevationMask or LandMask and a
        FlowMask, OR just a ChannelMask, as input.

        .. todo:: finish docstring

        Examples
        --------

        """
        if len(args) == 1:
            if isinstance(args[0], ChannelMask):
                _CM = args[0]
            else:
                raise TypeError(
                    'Expected ChannelMask.')
        elif len(args) == 2:
            for UnknownMask in args:
                if isinstance(UnknownMask, ElevationMask):
                    # make intermediate shoreline mask
                    _LM = LandMask.from_mask(UnknownMask, **kwargs)
                elif isinstance(UnknownMask, LandMask):
                    _LM = UnknownMask
                elif isinstance(UnknownMask, FlowMask):
                    _FM = UnknownMask
                else:
                    raise TypeError('type was %s' % type(UnknownMask))
            _CM = ChannelMask.from_mask(_LM, _FM)
        else:
            raise ValueError(
                'Must pass single ChannelMask, or two Masks to static '
                '`from_mask` for CenterlineMask.')

        # set up the empty shoreline mask
        _CntM = CenterlineMask(allow_empty=True)
        _CntM._set_shape_mask(_CM.shape)

        # compute the mask
        _CntM._compute_mask(_CM, **kwargs)
        return _CntM

    @staticmethod
    def from_array(_arr):
        """Create a CenterlineMask from an array.

        .. note::

            Instantiation with `from_array` will attempt to any data type
            (`dtype`) to boolean. This may have unexpected results. Convert
            your array to a boolean before using `from_array` to ensure the
            mask is created correctly.

        Parameters
        ----------
        _arr : :obj:`ndarray`
            The array with values to set as the mask. Can be any `dtype` but
            will be coerced to `boolean`.
        """
        # set directly
        raise NotImplementedError

    def __init__(self, *args, method='skeletonize', **kwargs):
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
        super().__init__('centerline', *args, **kwargs)

        self._method = method

        # temporary storage of args as needed for processing
        if self._input_flag is None:
            # do nothing, will need to call ._compute_mask later
            return

        elif self._input_flag == 'cube':
            raise NotImplementedError
            _tval = kwargs.pop('t', -1)
            _eta = args[0]['eta'][_tval, :, :]
            _flow = args[0]['velocity'][_tval, :, :]
            # need to convert these fields to proper masks

        elif self._input_flag == 'mask':
            # this pathway should allow someone to specify a combination of
            # elevation mask, landmask, and velocity mask or channelmask
            # directly, to make the new mask. This is basically an ambiguous
            # definition of the static methods.
            raise NotImplementedError

        elif self._input_flag == 'array':
            # first make a landmas
            _eta = args[0]
            _lm = LandMask(_eta, **kwargs)
            _flow = args[1]
            _fm = FlowMask(_flow, **kwargs)
            _CM = ChannelMask.from_mask(_lm, _fm)

        else:
            raise ValueError('Invalid _input_flag. Did you modify this attribute?')

        # save method type value to self
        self._method = method

        # compute the mask
        self._compute_mask(_CM)

    def _compute_mask(self, *args, **kwargs):
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
        if len(args) != 1:
            raise ValueError

        if isinstance(args[0], ChannelMask):
            cm_array = np.array(args[0]._mask, dtype=np.float)
        else:
            raise TypeError

        # check whether method was specified as a keyword arg to this method
        #    directly (this allows keyword spec for static methods)
        if 'method' in kwargs:
            self._method = kwargs.pop('method')

        # skimage.morphology.skeletonize() method
        if self.method == 'skeletonize':
            # for i in range(0, np.shape(self._mask)[0]):
            self._mask = morphology.skeletonize(cm_array)

        # rivamap based method
        if self.method == 'rivamap':
            # first check for import error
            try:
                from rivamap.singularity_index import applyMMSI as MMSI
                from rivamap.singularity_index import SingularityIndexFilters as SF
                from rivamap.delineate import extractCenterlines as eCL
            except ImportError:
                raise ImportError(
                    'You must install the optional dependency: rivamap, to '
                    'use the centerline extraction method.')
            except Exception as e:
                raise e

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
            self._mask = (nms_norm > self.nms_threshold)

    @property
    def method(self):
        """Method used to compute the mask.

        Returns
        -------
        method : :obj:`str`
            Method name as string.
        """
        return self._method


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
    @staticmethod
    def from_array(_arr):
        """Create a `GeometricMask` from an array.

        .. note::

            Instantiation with `from_array` will attempt to any data type
            (`dtype`) to boolean. This may have unexpected results. Convert
            your array to a boolean before using `from_array` to ensure the
            mask is created correctly.

        Parameters
        ----------
        _arr : :obj:`ndarray`
            The array with values to set as the mask. Can be any `dtype` but
            will be coerced to `boolean`.
        """
        # set directly
        raise NotImplementedError

    def __init__(self, *args, origin=None, **kwargs):
        """Initialize the GeometricMask.

        Initializing the geometric mask object requires information regarding
        the shape of the 2-D array of the region you wish to apply the mask
        to.

        .. important::

            The `GeometricMask` is initializes with all values set to
            ``True``; this is the opposite of all other `Mask` types, which
            are set to ``False`` during initialization.

        Parameters
        ----------
        *args :
            Various input arguments can be used to initialize the
            `GeometricMask`. Input may be a two-element `tuple`, specifying
            the array dimensions, or an `ndarray`, `Mask`, or `Cube` from
            which the shape is inferred.

        origin : `tuple`
            The "origin" of the domain. Usually this is the water/sediment
            inlet location. The `origin` is used as the mathematical origin
            point for computation setting the mask within the `GeometricMask`
            methods. If unspecified, it is inferred, based on the default
            configuration of a pyDeltaRCM model inlet.

        Examples
        --------
        """
        super().__init__('geometric', *args, **kwargs)

        # FOR GEOMETRIC, NEED START FROM ALL TRUE
        #   replace values from init immediately
        self._mask = np.ones(self.shape, dtype=np.bool)

        # pull the shape into components for convenience
        self._L, self._W = self.shape

        # set the origin from argument
        if origin is None:
            # try to infer it from the input type
            if self._input_flag == 'cube':
                raise NotImplementedError
                # get the value from CTR and L0 if meta present
            else:
                self._xc = 0
                self._yc = int(self._W / 2)
        elif isinstance(origin, tuple):
            # use the values in the tuple
            self._xc = origin[0]
            self._yc = origin[1]
        else:
            raise ValueError

    @property
    def xc(self):
        """x-coordinate of origin point."""
        return self._xc

    @property
    def yc(self):
        """y-coordinate of origin point."""
        return self._yc

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

        Examples
        --------

        .. todo:: write examples.

        """
        if (self._L / self._W) > 0.5:
            raise ValueError('Width of input array must exceed 2x length.')

        w = self._L if (self._L % 2 == 0) else self._L+1
        y, x = np.ogrid[0:self._W, -self._L:w]
        theta = np.arctan2(x, y) - theta1 + np.pi/2
        theta %= (2*np.pi)
        anglemask = theta <= (theta2-theta1)
        _, B = np.shape(anglemask)
        anglemap = anglemask[:self._L, int(B/2-self._W/2):int(B/2+self._W/2)]

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

        origin : :obj:`tuple`, optional
            Optionally specify an origin to use for computing the circle. Will
            use the `GeometricMask` origin, if not supplied.

        Examples
        --------

        .. todo:: write examples.

        """
        if origin is None:
            _xc = self._xc
            _yc = self._yc
        elif isinstance(origin, tuple):
            _xc = origin[0]
            _yc = origin[1]
        else:
            raise ValueError

        if rad2 is None:
            rad2 = np.max((self._L, self._W))

        yy, xx = np.meshgrid(range(self._W), range(self._L))
        # calculate array of distances from inlet
        raddist = np.sqrt((yy-_yc)**2 + (xx-_xc)**2)
        # identify points within radial bounds
        raddist = np.where(raddist >= rad1, raddist, 0)
        raddist = np.where(raddist <= rad2, raddist, 0)
        raddist = np.where(raddist == 0, raddist, 1)
        # make 3D to be consistent with mask
        raddist = np.reshape(raddist, [self._L, self._W])
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

        Examples
        --------

        .. todo:: write examples.

        """
        if ind2 is None:
            ind2 = self._L

        temp_mask = np.zeros_like(self._mask)
        temp_mask[ind1:ind2, :] = 1

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

        Examples
        --------

        .. todo:: write examples.

        """
        temp_mask = np.zeros_like(self._mask)
        if ind2 is None:
            w_ind = int(ind1/2)
            temp_mask[:, self._yc-w_ind:self._yc+w_ind+1] = 1
        else:
            temp_mask[:, ind1:ind2] = 1

        self._mask = self._mask * temp_mask

    def _compute_mask(self):
        """Does Nothing!"""
        pass
