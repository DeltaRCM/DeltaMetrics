import numpy as np

from scipy.spatial import ConvexHull
from shapely.geometry.polygon import Polygon
from skimage import morphology

import abc
import warnings

from numba import njit

from . import mask
from . import cube
from . import section as dm_section
from . import utils


class BasePlanform(abc.ABC):
    """Base planform object.

    Defines common attributes and methods of a planform object.

    This object should wrap around many of the functions available from
    :obj:`~deltametrics.mask` and :obj:`~deltametrics.mobility`.
    """

    def __init__(self, planform_type, *args, name=None):
        """Instantiate for subclasses of BasePlanform.

        The base class instantiation handles setting of the cooridnates of a
        Planform from the instantiating cube or xarray.

        Parameters
        ----------
        planform_type : :obj`str`
            The identifier for the *type* of Planform.

        CubeInstance : :obj:`~deltametrics.cube.Cube` subclass, optional
            Connect to this cube. No connection is made if cube is not
            provided.

        name : :obj:`str`, optional
            An optional name for the planform, helpful for maintaining and
            keeping track of multiple `Planform` objects of the same type.
            This is disctinct from the :obj:`planform_type`. The name is used
            internally if you use the `register_plan` method of a `Cube`.

        .. note::

            If no arguments are passed, an empty `Planform` not connected to
            any cube is returned. This cube may need to be manually connected
            to have any functionality (via the :meth:`connect` method); this
            need will depend on the type of `Planform`.
        """
        # begin unconnected
        self._x = None
        self._y = None
        self._shape = None
        self._variables = None
        self.cube = None
        self._composite_array = None

        self.planform_type = planform_type
        self._name = name

        # if len(args) != 1:
        #     raise ValueError('Expected single positional argument to \
        #                      %s instantiation.'
        #                      % type(self))

        if len(args) > 0:
            if issubclass(type(args[0]), cube.BaseCube):
                self.connect(args[0])
            elif utils.is_ndarray_or_xarray(args[0]):
                # use first argument as an array to get shape
                self._shape = args[0].shape

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
        self._shape = self.cube.shape[1:]

    @property
    def name(self):
        """Planform name.

        Helpful to differentiate multiple Planforms.
        """
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
        """Planform shape.
        """
        return self._shape

    @property
    def composite_array(self):
        """Array to extract a contour from when mask-making.

        This is the array that a contour is extracted from using some threshold
        value when making land and shoreline masks. This should just be an
        alias for either the `sea_angles` or the `meanimage` arrays from the
        OAM and MPM respectively.
        """
        return self._composite_array


class OpeningAnglePlanform(BasePlanform):
    """Planform for handling the Shaw Opening Angle Method.

    This `Planform` (called `OAP` for short) is a wrapper/handler for the
    input and output from the :func:`shaw_opening_angle_method`. The `OAP` is a
    convenient way to manage extraction of a shoreline or a delta topset area.

    Moreover, the `OAP` can be used as the input for :doc:`many types of
    Mask </reference/mask/index>` objects, so it is often computationally
    advantageous to compute this `Planform` once, and then use it to create
    many different types of masks.

    Examples
    --------
    Instantiate the `OpeningAnglePlanform` from an **inverted** binary mask of
    elevation data (i.e., from an :obj:`~deltametrics.mask.ElevationMask`).

    Note that the below example is the most verbose method for creating the
    `OAP`. Consider available static methods.

    .. plot::
        :context: reset
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> _EM = dm.mask.ElevationMask(
        ...     golfcube['eta'][-1, :, :],
        ...     elevation_threshold=0)

        # extract a mask of area below sea level as the
        #   inverse of the ElevationMask
        >>> _below_mask = ~(_EM.mask)

        >>> OAP = dm.plan.OpeningAnglePlanform(_below_mask)

    The OAP stores information computed from the
    :func:`shaw_opening_angle_method`. See the two properties of the OAP
    :obj:`below_mask` and :obj:`sea_angles`.

    .. plot::
        :context:

        fig, ax = plt.subplots(1, 3, figsize=(10, 4))
        golfcube.show_plan('eta', t=-1, ax=ax[0])
        im1 = ax[1].imshow(OAP.below_mask,
                           cmap='Greys_r', origin='lower')
        im2 = ax[2].imshow(OAP.sea_angles,
                           cmap='jet', origin='lower')
        dm.plot.append_colorbar(im2, ax=ax[2])
        ax[0].set_title('input elevation data')
        ax[1].set_title('OAP.below_mask')
        ax[2].set_title('OAP.sea_angles')
        for i in range(1, 3):
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    """

    @staticmethod
    def from_arrays(*args):
        """Create directly from arrays.

        .. warning:: not implemented.
        """
        raise NotImplementedError

    @staticmethod
    def from_elevation_data(elevation_data, **kwargs):
        """Create an `OpeningAnglePlanform` from elevation data.

        This process creates an ElevationMask from the input elevation array,
        and proceeds to make the OAP from the below sea level mask.

        .. note::

            Keyword arguments are passed to the `ElevationMask` *and* to the
            `OpeningAnglePlanform`, and thus passed to
            :func:`shaw_opening_angle_method`.

        .. important::

            The `elevation_threshold` argument is implicitly required in this
            method, because it is required to instantiate an
            :obj:`ElevationMask` from elevation data.

        Parameters
        ----------
        elevation_data : :obj:`ndarray`
            The elevation data to create the `ElevationMask` that is in
            turn used to create the `OpeningAnglePlanform`.

        Examples
        --------

        .. doctest::

            >>> golfcube = dm.sample_data.golf()

            >>> OAP = dm.plan.OpeningAnglePlanform.from_elevation_data(
            ...     golfcube['eta'][-1, :, :],
            ...     elevation_threshold=0)
        """
        # make a temporary mask
        _em = mask.ElevationMask(
            elevation_data, **kwargs)

        # invert the mask for the below sea level area
        _below_mask = ~(_em.mask)

        # compute from __init__ pathway
        return OpeningAnglePlanform(_below_mask, **kwargs)

    @staticmethod
    def from_ElevationMask(ElevationMask, **kwargs):
        """Create an `OpeningAnglePlanform` from an `ElevationMask`.

        .. note::

            Keyword arguments are passed to the `OpeningAnglePlanform`, and
            thus passed to :func:`shaw_opening_angle_method`.

        Parameters
        ----------
        ElevationMask : :obj:`~deltametrics.mask.ElevationMask`
            The :obj:`ElevationMask` to be used to create the
            `OpeningAnglePlanform`.

        Examples
        --------

        .. doctest::

            >>> golfcube = dm.sample_data.golf()
            >>> _EM = dm.mask.ElevationMask(
            ...     golfcube['eta'][-1, :, :],
            ...     elevation_threshold=0)

            >>> OAP = dm.plan.OpeningAnglePlanform.from_ElevationMask(
            ...     _EM)
        """
        if not isinstance(ElevationMask, mask.ElevationMask):
            raise TypeError('Must be type: ElevationMask.')

        # invert the mask for the below sea level area
        _below_mask = ~(ElevationMask.mask)

        # compute from __init__ pathway
        return OpeningAnglePlanform(_below_mask)

    @staticmethod
    def from_mask(UnknownMask, **kwargs):
        """Wraps :obj:`from_ElevationMask`.
        """
        return OpeningAnglePlanform.from_ElevationMask(
                UnknownMask, **kwargs)

    def __init__(self, *args, **kwargs):
        """Init.

        EXPECTS A BINARY OCEAN MASK AS THE INPUT!

        .. note:: needs docstring.

        """
        super().__init__('opening angle', *args)
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
                raise ValueError(
                    'Expected 1 input, got 0.')
        if not (len(args) == 1):
            raise ValueError(
                'Expected 1 input, got %s.' % str(len(args)))

        # process the argument to the omask needed for Shaw OAM
        if utils.is_ndarray_or_xarray(args[0]):
            _arr = args[0]
            # check that is boolean or integer binary
            if (_arr.dtype == bool):
                _below_mask = _arr
            elif (_arr.dtype == int):
                if np.all(np.logical_or(_arr == 0, _arr == 1)):
                    _below_mask = _arr
                else:
                    ValueError(
                        'The input was an integer array, but some elements in '
                        'the array were not 0 or 1.')
            else:
                raise TypeError(
                    'The input was not an integer or boolean array, but was '
                    '{0}. If you are trying to instantiate an OAP from '
                    'elevation data directly, see static method '
                    '`OpeningAnglePlanform.from_elevation_data`.')
        elif issubclass(type(args[0]), cube.BaseCube):
            raise NotImplementedError(
                'Instantiation from a Cube is not yet implemented.')
        else:
            # bad type supplied as argument
            raise TypeError('Invalid type for argument.')

        self._shape = _below_mask.shape

        self._compute_from_below_mask(_below_mask, **kwargs)

    def _compute_from_below_mask(self, below_mask, **kwargs):
        """Method for actual computation of the arrays.

        Parameters
        ----------
        below_mask
            The binarized array of values that should be considered as the
            ocean pixels.

        **kwargs
            Passed to :func:`shaw_opening_angle_method`.
        """

        sea_angles = np.zeros(self._shape)

        # check if there is any *land*
        if np.any(below_mask == 0):

            # need to convert type to integer
            below_mask = below_mask.astype(int)

            # pull out the shaw oam keywords
            shaw_kwargs = {}
            if 'numviews' in kwargs:
                shaw_kwargs['numviews'] = kwargs.pop('numviews')

            # pixels present in the mask
            shoreangles, seaangles = shaw_opening_angle_method(
                below_mask, **shaw_kwargs)

            # translate flat seaangles values to the shoreline image
            #  this is a good target for optimization (return reshaped?)
            flat_inds = list(map(
                lambda x: np.ravel_multi_index(x, sea_angles.shape),
                seaangles[:2, :].T.astype(int)))
            sea_angles.flat[flat_inds] = seaangles[-1, :]

        # assign shore_image to the mask object with proper size
        self._sea_angles = sea_angles

        # properly assign the oceanmap to the self.below_mask
        #   set it to be bool regardless of input type
        self._below_mask = below_mask.astype(bool)

    @property
    def sea_angles(self):
        """Maximum opening angle view of the sea from a pixel.

        See figure in main docstring for visual example.
        """
        return self._sea_angles

    @property
    def below_mask(self):
        """Mask for below sea level pixels.

        This is the starting point for the Opening Angle Method solution.

        See figure in main docstring for visual example.
        """
        return self._below_mask

    @property
    def composite_array(self):
        """Alias sea angles."""
        return self._sea_angles


class MorphologicalPlanform(BasePlanform):
    """Planform for handling the morphological method.

    .. todo::

        Expand docstring

    """

    @staticmethod
    def from_elevation_data(elevation_data, max_disk, **kwargs):
        """Create a `MorphologicalPlanform` from elevation data.

        Creates an ElevationMask from the input elevation array that is used
        to create the MP.

        .. note::

            Information about keyword arguments

        .. important::

            The `elevation_threshold` argument is implicitly required in this
            method, because it is required to instantiate an
            :obj:`ElevationMask` from elevation data.

        Parameters
        ----------
        elevation_data : :obj:`ndarray`
            The elevation data to create the `ElevationMask` that is in
            turn used to create the `MorphologicalPlanform`.

        max_disk : int
            Maximum disk size to use for the morphological operations.

        Examples
        --------

        .. doctest::

            >>> golfcube = dm.sample_data.golf()

            >>> MP = dm.plan.MorphologicalPlanform.from_elevation_data(
            ...     golfcube['eta'][-1, :, :],
            ...     elevation_threshold=0,
                    max_disk=3)
        """
        # make a temporary mask
        _em = mask.ElevationMask(
            elevation_data, **kwargs)

        # compute from __init__ pathway
        return MorphologicalPlanform(_em, max_disk, **kwargs)

    @staticmethod
    def from_mask(UnknownMask, maxdisk, **kwargs):
        """Static method for creating a MorphologicalPlanform from a mask."""
        return MorphologicalPlanform(UnknownMask, maxdisk, **kwargs)

    def __init__(self, *args, **kwargs):
        """Initialize the MP.

        Expects first argument to be either an ElevationMask, or an array that
        represents some sort of elevation mask or land area for the delta.

        Second argument should be the inlet width (# pixels), if a cube is
        connected then this will be pulled from the cube directly.

        Method should work if a landmask is provided too, the morphological
        operations may just do less.

        .. todo::

            Improve docstring.

        """
        super().__init__('morphological method', *args)
        self._shape = None
        self._elevation_mask = None
        self._max_disk = None

        # check for input or allowable emptiness
        if (len(args) == 0):
            _allow_empty = kwargs.pop('allow_empty', False)
            if _allow_empty:
                # do nothing and return partially instantiated object
                return
            else:
                raise ValueError(
                    'Expected at least 1 input, got 0.')
        # assign first argument to attribute of self
        if isinstance(args[0], mask.BaseMask):
            self._elevation_mask = args[0]
        elif utils.is_ndarray_or_xarray(args[0]):
            self._elevation_mask = args[0]
        else:
            raise TypeError(
                'Type of first argument is unrecognized or unsupported')
        # see if the inlet width is provided, if not see if cube is avail
        if (len(args) > 1):
            if isinstance(args[1], (int, float)):
                self._max_disk = int(args[1])
            else:
                raise TypeError(
                'Expected single number to set max inlet size, got something '
                'else instead.')
        elif isinstance(self.cube, cube.BaseCube):
            try:
                self._max_disk = self.cube.meta['N0'].data
            except Exception:
                raise TypeError(
                'Data cube does not contain metadata, specify the inlet size')
        else:
            raise TypeError(
            'Something went wrong. This error message should be better.')

        self._shape = self._elevation_mask.shape

        # run the computation
        all_images, mean_image = morphological_closing_method(
            self._elevation_mask, biggestdisk=self._max_disk)

        # assign arrays to object
        self._mean_image = mean_image
        self._all_images = all_images

    @property
    def mean_image(self):
        """Average of all binary closing arrays."""
        return self._mean_image

    @property
    def all_images(self):
        """3-D array of all binary closed arrays."""
        return self._all_images

    @property
    def composite_array(self):
        """Alias the mean image."""
        return self._mean_image

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
    the roughness later. Here, we use the `elevation_offset` parameter (passed
    to :obj:`~deltametrics.mask.ElevationMask`) to better capture the
    topography of the `pyDeltaRCM` model results.

    .. plot::
        :include-source:
        :context: reset

        golf = dm.sample_data.golf()

        # early in model run
        lm0 = dm.mask.LandMask(
            golf['eta'][15, :, :],
            elevation_threshold=0,
            elevation_offset=-0.5)
        sm0 = dm.mask.ShorelineMask(
            golf['eta'][15, :, :],
            elevation_threshold=0,
            elevation_offset=-0.5)

        # late in model run
        lm1 = dm.mask.LandMask(
            golf['eta'][-1, :, :],
            elevation_threshold=0,
            elevation_offset=-0.5)
        sm1 = dm.mask.ShorelineMask(
            golf['eta'][-1, :, :],
            elevation_threshold=0,
            elevation_offset=-0.5)

    Let's take a quick peek at the masks that we have created.

    .. plot::
        :include-source:
        :context:

        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        lm0.show(ax=ax[0])
        sm0.show(ax=ax[1])
        plt.show()

    In order for these masks to work as expected in the shoreline roughness
    computation, we need to modify the mask values slightly, to remove the
    land-water boundary that is not really a part of the delta. We use the
    :meth:`~deltametrics.mask.BaseMask.trim_mask` method to trim a mask.

    .. plot::
        :include-source:
        :context: close-figs

        lm0.trim_mask(length=golf.meta['L0'].data+1)
        sm0.trim_mask(length=golf.meta['L0'].data+1)
        lm1.trim_mask(length=golf.meta['L0'].data+1)
        sm1.trim_mask(length=golf.meta['L0'].data+1)

        fig, ax = plt.subplots(1, 2, figsize=(8, 3))
        lm0.show(ax=ax[0])
        sm0.show(ax=ax[1])
        plt.show()

    And now, we can proceed with the calculation.

    .. plot::
        :include-source:
        :context: close-figs

        # compute roughnesses
        rgh0 = dm.plan.compute_shoreline_roughness(sm0, lm0)
        rgh1 = dm.plan.compute_shoreline_roughness(sm1, lm1)

        # make the plot
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        golf.show_plan('eta', t=15, ax=ax[0])
        ax[0].set_title('roughness = {:.2f}'.format(rgh0))
        golf.show_plan('eta', t=-1, ax=ax[1])
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
        Shoreline mask. Can be a :obj:`~deltametrics.mask.ShorelineMask`
        object, or a binarized array.

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
    the length later. Here, we use the `elevation_offset` parameter (passed to
    :obj:`~deltametrics.mask.ElevationMask`) to better capture the topography
    of the `pyDeltaRCM` model results.

    .. plot::
        :include-source:

        golf = dm.sample_data.golf()

        # early in model run
        sm0 = dm.mask.ShorelineMask(
            golf['eta'][15, :, :],
            elevation_threshold=0,
            elevation_offset=-0.5)

        # late in model run
        sm1 = dm.mask.ShorelineMask(
            golf['eta'][-1, :, :],
            elevation_threshold=0,
            elevation_offset=-0.5)

        # compute lengths
        len0 = dm.plan.compute_shoreline_length(sm0)
        len1, line1 = dm.plan.compute_shoreline_length(sm1, return_line=True)

        # make the plot
        fig, ax = plt.subplots(1, 2, figsize=(6, 3))
        golf.show_plan('eta', t=15, ax=ax[0])
        ax[0].set_title('length = {:.2f}'.format(len0))
        golf.show_plan('eta', t=-1, ax=ax[1])
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
    hit_pts = np.zeros(len(_x), dtype=bool)
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


def compute_shoreline_distance(shore_mask, origin=[0, 0],
                               return_distances=False):
    """Compute mean and stddev distance from the delta apex to the shoreline.

    Algorithm computes the mean distance from the delta apex/origin to all
    shoreline points.

    .. important::

        This calculation is subtly different than the "mean delta radius",
        because the measurements are not sampled evenly along the opening
        angle of the delta.

    .. note:: uses `np.nanmean` and `np.nanstd`.

    Parameters
    ----------
    shore_mask : :obj:`~deltametrics.mask.ShorelineMask`, :obj:`ndarray`
        Shoreline mask. Can be a :obj:`~deltametrics.mask.ShorelineMask`
        object, or a binarized array.

    origin : :obj:`list`, :obj:`np.ndarray`, optional
        Determines the location from where the distance to all shoreline
        points is computed.

    return_distances : :obj:`bool`
        Whether to return the sorted line as a second argument. If True, a
        ``Nx2`` array of x-y points is returned. Default is `False`.

    Returns
    -------
    mean : :obj:`float`
        Mean shoreline distance.

    stddev : :obj:`float`
        Standard deviation of shoreline distance.

    distances : :obj:`np.ndarray`
        If :obj:`return_distances` is `True`, then distance to each point
        along the shoreline is *also* returned as an array (i.e., 3 arguments
        are returned).

    Examples
    --------

    .. doctest::

        golf = dm.sample_data.golf()

        sm = dm.mask.ShorelineMask(
            golf['eta'][-1, :, :],
            elevation_threshold=0,
            elevation_offset=-0.5)

        # compute mean and stddev distance
        mean, stddev = dm.plan.compute_shoreline_distance(
            sm, origin=[golf.meta['CTR'].data, golf.meta['L0'].data])

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

    # determine the distances
    _dists = np.sqrt((_x - origin[0])**2 + (_y - origin[1])**2)

    if return_distances:
        return np.nanmean(_dists), np.nanstd(_dists), _dists
    else:
        return np.nanmean(_dists), np.nanstd(_dists)


@njit
def _compute_angles_between(c1, shoreandborder, Shallowsea, numviews):
    """Private helper for shaw_opening_angle_method.

    Good target for code style, organization, and optimization.
    """
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
        Defines the number of largest angles to consider for the opening angle
        map for each pixel. Default is 3, based on [1]_.

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
    In = In.astype(bool)

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
    Deepsea[-1, :] = 180.  # 180 is a background value for waves1s later

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


def _custom_closing(img, disksize):
    """Private function for the binary closing."""
    _changed = np.infty
    disk = morphology.disk(disksize)
    _iter = 0  # count number of closings
    while (_changed != 0) and (_iter < 1000):
        _iter += 1
        print(_iter)
        _newimg = morphology.binary_closing(img, selem=disk)
        _changed = np.sum(_newimg.astype(float)-img.astype(float))
        _closed = _newimg
    return _closed


def morphological_closing_method(elevationmask, biggestdisk=None):
    """Compute an average morphological map from an image,

    Applies a morphological closing to the input image in a manner
    similar to / inspired by [1]_ for rapid identification of a shoreline.

    This *function* takes an image, and performs a morphological closing for
    a set of disk sizes up from 0 up to the parameter `biggestdisk`.

    .. [1] Geleynse, N., et al. "Characterization of river delta shorelines."
       Geophysical research letters 39.17 (2012).

    Parameters
    ----------
    elevationmask : :obj:`~deltametrics.mask.ElevationMask` or
                    :obj:`ndarray` or :obj:`xarray`
        Binary image that the morpholigical closing is performed upon.
        This is expected to be something like an elevation mask, although it
        doesn't have to be.

    biggestdisk : int, optional
        Defines the largest disk size to use for the binary closing method.
        The method starts 0 and iterates up to a disk size of biggestdisk.

    Returns
    -------
    imageset : ndarray
        3-D array of shape n-x-y where n is the number of different disk
        kernels used in the method. n = biggestdisk + 1

    meanimage : ndarray
        2-D array of shape x-y of the mean of imageset taken over the first
        axis. This approximates the `sea_angles` attribute of the OAM method.
    """
    # coerce input image into 2-d ndarray
    if isinstance(elevationmask, mask.BaseMask):
        emsk = np.array(elevationmask.mask)
    elif utils.is_ndarray_or_xarray(elevationmask):
        emsk = np.array(elevationmask)
    else:
        raise TypeError(
            'Input for `elevationmask` was unrecognized type: {}.'.format(
                type(elevationmask)))

    # check biggestdisk
    if biggestdisk is None:
        biggestdisk = 1
    elif biggestdisk <= 0:
        biggestdisk = 1

    # loop through and do binary closing for each disk size up to biggestdisk
    imageset = np.zeros((biggestdisk+1, emsk.shape[0], emsk.shape[1]))
    for i in range(biggestdisk+1):
        imageset[i, ...] = _custom_closing(emsk, i)

    return imageset, imageset.mean(axis=0)


def compute_channel_width(channelmask, section=None, return_widths=False):
    """Compute channel width from a mask and section.

    Compute the width of channels identified in a ChannelMask along a section.
    This function identifies the individual channels that are crossed by the
    section and computes width of each channel as the along-section distance.

    In essence, this processing implicitly assumes that the section cuts each
    channel perpendicularly. We therefore recommend using this function with
    a `~dm.section.CircularSection` type, unless you know what you are doing.
    By default, only the mean and standard deviation are returned, but the
    list of widths can be returned with `return_widths=True`.

    .. note::

        If a `numpy` array is passed for :obj:`section`, then the distance
        between points along the section is assumed to be `==1`.

    Parameters
    ----------
    channelmask : :obj:`~deltametrics.mask.ChannelMask` or :obj:`ndarray`
        The channel mask (i.e., should be binary) to compute channel widths
        from.

    section : :obj:`~deltametrics.section.BaseSection` subclass, or :obj:`ndarray`
        The section along which to compute channel widths. If a `Section` type
        is passed, the `.trace` attribute will be used to query the
        `ChannelMask` and determine widths. Otherwise, an `Nx2` array can be
        passed, which specified the x-y coordinate pairs to use as the
        trace.

    return_widths : bool, optional
        Whether to return (as third argument) a list of channel widths.
        Default is false (do not return list).

    Returns
    -------
    mean : float
        Mean of measured widths.

    stddev : float
        Standard deviation of measured widths.

    widths : list
        List of width measurements. Returned only if `return_widths=True`.

    Examples
    --------

    .. plot::
        :include-source:

        # set up the cube, mask, and section
        golf = dm.sample_data.golf()
        cm = dm.mask.ChannelMask(
            golf['eta'][-1, :, :],
            golf['velocity'][-1, :, :],
            elevation_threshold=0,
            flow_threshold=0.3)
        sec = dm.section.CircularSection(golf, radius=40)

        # compute the metric
        m, s, w = dm.plan.compute_channel_width(
            cm, section=sec, return_widths=True)

        fig, ax = plt.subplots()
        cm.show(ax=ax)
        sec.show_trace('r-', ax=ax)
        ax.set_title(f'mean: {m:.2f}; stddev: {s:.2f}')
        plt.show()
    """
    if not (section is None):
        if issubclass(type(section), dm_section.BaseSection):
            section_trace = section.trace
            section_coord = section._s
        elif isinstance(section, np.ndarray):
            section_trace = section
            section_coord = np.arange(len(section))
    else:
        # create one by default based on the channelmask?
        raise NotImplementedError()

    # check that the section trace is a valid shape
    #   todo...

    if utils.is_ndarray_or_xarray(channelmask):
        pass
    elif isinstance(channelmask, mask.ChannelMask):
        channelmask = np.array(channelmask.mask)
    else:
        raise TypeError(
            'Input for `channelmask` was wrong type: {}.'.format(
                type(channelmask)))

    _channelseries = channelmask[section_trace[:, 1],
                                 section_trace[:, 0]].astype(int)

    # compute the metrics
    _padchannelseries = np.pad(_channelseries, (1,), 'constant',
                               constant_values=(False)).astype(int)
    _channelseries_diff = _padchannelseries[1:] - _padchannelseries[:-1]
    _channelstarts = np.where(_channelseries_diff == 1)[0]
    _channelends = np.where(_channelseries_diff == -1)[0]

    _channelwidths = section_coord[_channelends-1] - section_coord[_channelstarts-1]

    _m, _s = np.nanmean(_channelwidths), np.nanstd(_channelwidths)
    if return_widths:
        return _m, _s, _channelwidths
    else:
        return _m, _s


def compute_channel_depth(channelmask, depth, section=None,
                          depth_type='thalweg', return_depths=False):
    """Compute channel depth from a mask and section.

    Compute the depth of channels identified in a ChannelMask along a section.
    This function identifies the individual channels that are crossed by the
    section and *computes depth of each*. The depths are then treated as
    samples for aggregating statistics in the return.

    By default, only the mean and standard deviation are returned, but the
    list of depths can be returned with `return_depths=True`.

    .. note::

        If a `numpy` array is passed for :obj:`section`, then the distance
        between points along the section is assumed to be `==1`.

    Parameters
    ----------
    channelmask : :obj:`~deltametrics.mask.ChannelMask` or :obj:`ndarray`
        The channel mask (i.e., should be binary) to compute channel depths
        from.

    depth : `xarray` or `ndarray`
        The depth field corresponding to the channelmask array.

    section : :obj:`~deltametrics.section.BaseSection` subclass, or :obj:`ndarray`
        The section along which to compute channel depths. If a `Section` type
        is passed, the `.trace` attribute will be used to query the
        `ChannelMask` and determine depths. Otherwise, an `Nx2` array can be
        passed, which specified the x-y coordinate pairs to use as the
        trace.

    depth_type : :obj:`str`
        Flag indicating how to compute the depth of *each* channel
        (i.e., before aggregating). Valid flags are `'thalweg'`(default) and
        `'mean'`.

    return_depths : bool, optional
        Whether to return (as third argument) a list of channel depths.
        Default is false (do not return list).

    Returns
    -------
    mean : float
        Mean of measured depths.

    stddev : float
        Standard deviation of measured depths.

    depths : list
        List of depth measurements. Returned only if `return_depths=True`.
    """
    if not (section is None):
        if issubclass(type(section), dm_section.BaseSection):
            section_trace = section.trace
            section_coord = section._s
        elif isinstance(section, np.ndarray):
            section_trace = section
            section_coord = np.arange(len(section))
    else:
        # create one by default based on the channelmask?
        raise NotImplementedError()

    # check that the section trace is a valid shape
    #   todo...

    if utils.is_ndarray_or_xarray(channelmask):
        pass
    elif isinstance(channelmask, mask.ChannelMask):
        channelmask = np.array(channelmask.mask)
    else:
        raise TypeError(
            'Input for `channelmask` was wrong type: {}.'.format(
                type(channelmask)))

    # need to get the channel starts and ends
    _channelseries = channelmask[section_trace[:, 1],
                                 section_trace[:, 0]].astype(int)
    _padchannelseries = np.pad(_channelseries, (1,), 'constant',
                               constant_values=(False)).astype(int)
    _channelseries_diff = _padchannelseries[1:] - _padchannelseries[:-1]
    _channelstarts = np.where(_channelseries_diff == 1)[0]
    _channelends = np.where(_channelseries_diff == -1)[0]

    _channelwidths = section_coord[_channelends-1] - section_coord[_channelstarts-1]

    # get the depth array along the section
    _depthslice = np.copy(depth)
    _depthseries = _depthslice[section_trace[:, 1], section_trace[:, 0]]

    # for depth and area of channels, we loop through each discrete channel
    _channel_depth_means = np.full(len(_channelwidths), np.nan)
    _channel_depth_thalweg = np.full(len(_channelwidths), np.nan)
    # _channel_depth_area = np.full(len(_channelwidths), np.nan)
    for k in np.arange(len(_channelwidths)):
        # extract the depths for the kth channel
        _kth_channel_depths = _depthseries[_channelstarts[k]:_channelends[k]]

        # compute the mean depth of kth channel and the thalweg of this channel
        _channel_depth_means[k] = np.nanmean(_kth_channel_depths)

        # compute the max depth, aka the thalweg
        _channel_depth_thalweg[k] = np.max(_kth_channel_depths)

    if depth_type == 'thalweg':
        _channel_depth_list = _channel_depth_thalweg
    elif depth_type == 'mean':
        _channel_depth_list = _channel_depth_means
    else:
        raise ValueError(
            'Invalid argument to `depth_type` {}'.format(
                str(depth_type)))

    _m, _s = np.mean(_channel_depth_list), np.std(_channel_depth_list)
    if return_depths:
        return _m, _s, _channel_depth_list
    else:
        return _m, _s
