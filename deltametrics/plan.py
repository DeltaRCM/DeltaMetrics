import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scipy.spatial import ConvexHull
from shapely.geometry.polygon import Polygon
from skimage import morphology

import abc
import warnings

from numba import njit

from . import mask
from . import cube
from . import section as dm_section
from . import plot
from . import utils


class BasePlanform(abc.ABC):
    """Base planform object.

    Defines common attributes and methods of all planform objects.
    """

    def __init__(self, planform_type, *args, name=None):
        """Instantiate for subclasses of BasePlanform.

        The base class instantiation handles setting the `name` attribute of
        the `Planform`, and defines the internal plotting routine
        via :obj:`_show`.

        Parameters
        ----------
        planform_type : :obj`str`
            String identifying the *type* of `Planform` being instantiated.

        *args
            Arbitrary arguments, passed from the subclass, not used here.

        name : :obj:`str`, optional
            An optional name for the planform, helpful for maintaining and
            keeping track of multiple `Planform` objects of the same type.
            This is disctinct from the :obj:`planform_type`. The name is used
            internally if you use the :obj:`register_planform` method of a
            `Cube`.
        """
        # begin unconnected
        self._shape = None
        self._variables = None

        self.planform_type = planform_type
        self._name = name

    @property
    def name(self):
        """Planform name.

        Helpful to differentiate multiple `Planform` objects.
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

    def _show(self, field, varinfo, **kwargs):
        """Internal method for showing a planform.

        Each planform may implement it's own method to determine what field to
        show when called, and different calling options.

        Parameters
        ----------
        field : :obj:`DataArray`
            The data to show.

        varinfo : :obj:`VariableInfo`
            A :obj:`VariableInfo` instance describing how to color `field`.

        **kwargs
            Acceptable kwargs are `ax`, `title`, `ticks`, `colorbar`,
            `colorbar_label`. See description for `DataPlanform.show` for
            more information.
        """
        # process arguments and inputs
        ax = kwargs.pop('ax', None)
        title = kwargs.pop('title', None)
        ticks = kwargs.pop('ticks', False)
        colorbar = kwargs.pop('colorbar', True)
        colorbar_label = kwargs.pop('colorbar_label', False)

        if not ax:
            ax = plt.gca()

        # get the extent as arbitrary dimensions
        d0, d1 = field.dims
        d0_arr, d1_arr = field[d0], field[d1]
        _extent = [d1_arr[0],                  # dim1, 0
                   d1_arr[-1] + d1_arr[1],     # dim1, end + dx
                   d0_arr[-1] + d0_arr[1],     # dim0, end + dx
                   d0_arr[0]]                  # dim0, 0

        im = ax.imshow(field,
                       cmap=varinfo.cmap,
                       norm=varinfo.norm,
                       vmin=varinfo.vmin,
                       vmax=varinfo.vmax,
                       extent=_extent)

        if colorbar:
            cb = plot.append_colorbar(im, ax)
            if colorbar_label:
                _colorbar_label = \
                    varinfo.label if (colorbar_label is True) \
                    else str(colorbar_label)  # use custom if passed
                cb.ax.set_ylabel(_colorbar_label, rotation=-90, va="bottom")

        if not ticks:
            ax.set_xticks([], minor=[])
            ax.set_yticks([], minor=[])
        if title:
            ax.set_title(str(title))

        return im


class Planform(BasePlanform):
    """Basic Planform object.

    This class is used to slice the `Cube` along the `dim0` axis. The object
    is akin to the various `Section` classes, but there is only the one way
    to slice as a Planform.
    """

    def __init__(self, *args, z=None, t=None, idx=None, **kwargs):
        """
        Identify coordinate defining the planform.

        Parameters
        ----------

        CubeInstance : :obj:`~deltametrics.cube.BaseCube` subclass, optional
            Connect to this cube. No connection is made if cube is not
            provided.

        z : :obj:`float`, optional

        t : :obj:`float`, optional

        idx : :obj:`int`, optional

        Notes
        -----

        If no positional arguments are passed, an empty `Planform` not
        connected to any cube is returned. This cube may need to be manually
        connected to have any functionality (via the :meth:`connect` method);
        this need will depend on the type of `Planform`.
        """
        if (not (z is None)) and (not (idx is None)):
            raise TypeError('Cannot specify both `z` and `idx`.')
        if (not (t is None)) and (not (idx is None)):
            raise TypeError('Cannot specify both `t` and `idx`.')
        if (not (z is None)) and (not (t is None)):
            raise TypeError('Cannot specify both `z` and `t`.')

        self.cube = None
        self._dim0_idx = None

        self._input_z = z
        self._input_t = t
        self._input_idx = idx

        super().__init__('data', *args, **kwargs)

        if len(args) > 0:
            self.connect(args[0])
        else:
            pass

    @property
    def variables(self):
        """List of variables.
        """
        return self._variables

    @property
    def idx(self):
        """Index into underlying Cube along axis 0.
        """
        return self._dim0_idx

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

        self._compute_planform_coords()

    def _compute_planform_coords(self):
        """Should calculate vertical coordinate of the section.

        Sets the value ``self._dim0_idx`` according to
        the algorithm of a `Planform` initialization.

        .. warning::

            When implementing a new planform type, be sure that
            ``self._dim0_idx`` is a  *one-dimensional array*, or you will get
            an improperly shaped Planform array in return.
        """

        # determine the index along dim0 to slice cube
        if (not (self._input_z is None)) or (not (self._input_t is None)):
            # z an t are treated the same internally, and either will be
            # silently used  to interpolate the dim0 coordinates to find the
            # nearest index
            dim0_val = self._input_z or self._input_t
            self._dim0_idx = np.argmin(np.abs(
                np.array(self.cube.dim0_coords) - dim0_val))
        else:
            # then idx must have been given
            self._dim0_idx = self._input_idx

    def __getitem__(self, var):
        """Get a slice of the planform.

        Slicing the planform instance creates an `xarray` `DataArray` instance
        from data for variable ``var``.

        .. note:: We only support slicing by string.

        Parameters
        ----------
        var : :obj:`str`
            Which variable to slice.

        Returns
        -------
        data : :obj:`DataArray`
            The undelrying data returned as an xarray `DataArray`, maintaining
            coordinates.
        """
        if isinstance(self.cube, cube.DataCube):
            _xrDA = self.cube[var][self._dim0_idx, :, :]
            _xrDA.attrs = {'slicetype': 'data_planform',
                           'knows_stratigraphy': self.cube._knows_stratigraphy,
                           'knows_spacetime': True}
            if self.cube._knows_stratigraphy:
                _xrDA.strat.add_information(
                    _psvd_mask=self.cube.strat_attr.psvd_idx[self._dim0_idx, :, :],  # noqa: E501
                    _strat_attr=self.cube.strat_attr(
                        'planform', self._dim0_idx, None))
            return _xrDA
        elif isinstance(self.cube, cube.StratigraphyCube):
            _xrDA = self.cube[var][self._dim0_idx, :, :]
            _xrDA.attrs = {'slicetype': 'stratigraphy_planform',
                           'knows_stratigraphy': True,
                           'knows_spacetime': False}
            return _xrDA
        elif (self.cube is None):
            raise AttributeError(
                'No cube connected. Are you sure you ran `.connect()`?')
        else:
            raise TypeError('Unknown Cube type encountered: %s'
                            % type(self.cube))

    def show(self, var, ax=None, title=None, ticks=False,
             colorbar=True, colorbar_label=False):
        """Show the planform.

        Method enumerates convenient routines for visualizing planform data
        and slices of stratigraphy.

        Parameters
        ----------
        var : :obj:`str`
            Which attribute to show. Can be a string for a named `Cube`
            attribute.

        label : :obj:`bool`, `str`, optional
            Display a label of the variable name on the plot. Default is
            False, display nothing. If ``label=True``, the label name from the
            :obj:`~deltametrics.plot.VariableSet` is used. Other arguments are
            attempted to coerce to `str`, and the literal is diplayed.

        colorbar : :obj:`bool`, optional
            Whether a colorbar is appended to the axis.

        colorbar_label : :obj:`bool`, `str`, optional
            Display a label of the variable name along the colorbar. Default is
            False, display nothing. If ``label=True``, the label name from the
            :obj:`~deltametrics.plot.VariableSet` is used. Other arguments are
            attempted to coerce to `str`, and the literal is diplayed.

        ax : :obj:`~matplotlib.pyplot.Axes` object, optional
            A `matplotlib` `Axes` object to plot the section. Optional; if not
            provided, a call is made to ``plt.gca()`` to get the current (or
            create a new) `Axes` object.

        Examples
        --------
        Display the `eta` and `velocity` planform of a DataCube.

        .. plot::
            :include-source:

            >>> golfcube = dm.sample_data.golf()
            >>> planform = dm.plan.Planform(golfcube, idx=70)
            ...
            >>> fig, ax = plt.subplots(1, 2)
            >>> planform.show('eta', ax=ax[0])
            >>> planform.show('velocity', ax=ax[1])
            >>> plt.show()
        """
        # process the planform attribute to a field
        _varinfo = self.cube.varset[var] if \
            issubclass(type(self.cube), cube.BaseCube) else \
            plot.VariableSet()[var]
        _field = self[var]

        # call the internal _show method
        im = self._show(
            _field, _varinfo,
            ax=ax, title=title, ticks=ticks,
            colorbar=colorbar, colorbar_label=colorbar_label)

        return im


class SpecialtyPlanform(BasePlanform):
    """A base class for All specialty planforms.

    .. hint:: All specialty planforms should subclass.

    Specialty planforms are planforms that hold some computation or attribute
    *about* some underlying data, rather than the actual data. As a general
    rule, anything that is not a DataPlanform is a SpecialtyPlanform.

    This base class implements a slicing method (it slices the `data` field),
    and a `show` method for displaying the planform (it displays the `data`
    field).

    .. rubric:: Developer Notes

    All subclassing objects must implement:
      * a property named `data` that points to some field (i.e., an attribute
        of the planform) that best characterizes the Planform. For example,
        the OAP planform `data` property points to the `sea_angles` field.

    All subclassing objects should consider implementing:
      * the `show` method takes (optionally) a string argument specifying the
        field to display, which can match any attriute of the
        `SpecialtyPlanform`. If no argument is passed to `show`, the `data`
        field is displayed. A :obj:`VariableInfo` object
        `self._default_varinfo` is created on instantiating a subclass, which
        will be used to style the displayed field. You can add different
        `VariableInfo` objects with the name matching any other field of the
        planform to use that style instead; for example, OAP implements
        `self._sea_angles_varinfo`, which is used if the `sea_angles` field
        is specified to :meth:`show`.
      * The `self._default_varinfo` can be overwritten in a subclass
        (after ``super().__init__``) to style the `show` default field
        (`data`) a certain way. For example, OAP sets ``self._default_varinfo
        = self._sea_angles_varinfo``.
    """

    def __init__(self, planform_type, *args, **kwargs):
        """Initialize the SpecialtyPlanform.

        BaseClass, only called by subclassing methods. This `__init__` method
        calls the `BasePlanform.__init__`.

        Parameters
        ----------
        planform_type : :obj:`str`
            A string specifying the type of planform being created.

        *args
            Passed to `BasePlanform.__init__`.

        *kwargs
            Passed to `BasePlanform.__init__`.
        """
        super().__init__(planform_type, *args, **kwargs)

        self._default_varinfo = plot.VariableInfo(
            'data', label='data')

    @property
    @abc.abstractmethod
    def data(self):
        """The public data field.

        This attribute *must* be implemented as an alias to another attribute.
        The choice of field is up to the developer.
        """
        ...

    def __getitem__(self, slc):
        """Slice the planform.

        Implements basic slicing for `SpecialtyPlanform` by passing the `slc`
        to `self.data`. I.e., the returned slice is ``self.data[slc]``.
        """
        return self.data[slc]

    def show(self, var=None, ax=None, title=None, ticks=False,
             colorbar=True, colorbar_label=False):
        """Show the planform.

        Display a field of the planform, called by attribute name.

        Parameters
        ----------
        var : :obj:`str`
            Which field to show. Must be an attribute of the planform. `show`
            will look for another attribute describing
            the :obj:`VariableInfo` for that attribute named
            ``self._<var>_varinfo`` and use that to style the plot, if
            found. If this `VariableInfo` is not found, the default is used.

        label : :obj:`bool`, `str`, optional
            Display a label of the variable name on the plot. Default is
            False, display nothing. If ``label=True``, the label name from the
            :obj:`~deltametrics.plot.VariableSet` is used. Other arguments are
            attempted to coerce to `str`, and the literal is diplayed.

        colorbar : :obj:`bool`, optional
            Whether a colorbar is appended to the axis.

        colorbar_label : :obj:`bool`, `str`, optional
            Display a label of the variable name along the colorbar. Default is
            False, display nothing. If ``label=True``, the label name from the
            :obj:`~deltametrics.plot.VariableSet` is used. Other arguments are
            attempted to coerce to `str`, and the literal is diplayed.

        ax : :obj:`~matplotlib.pyplot.Axes` object, optional
            A `matplotlib` `Axes` object to plot the section. Optional; if not
            provided, a call is made to ``plt.gca()`` to get the current (or
            create a new) `Axes` object.
        """
        if (var is None):
            _varinfo = self._default_varinfo
            _field = self.data
        elif (isinstance(var, str)):
            _field = self.__getattribute__(var)  # will error if var not attr
            _expected_varinfo = '_' + var + '_varinfo'
            if hasattr(self, _expected_varinfo):
                _varinfo = self.__getattribute__(_expected_varinfo)
            else:
                _varinfo = self._default_varinfo
        else:
            raise TypeError('Bad value for `var`: {0}'.format(var))

        self._show(
            _field, _varinfo,
            ax=ax, title=title, ticks=ticks,
            colorbar=colorbar, colorbar_label=colorbar_label)


class OpeningAnglePlanform(SpecialtyPlanform):
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

        >>> # extract a mask of area below sea level as the
        >>> #   inverse of the ElevationMask
        >>> below_mask = ~(_EM.mask)

        >>> OAP = dm.plan.OpeningAnglePlanform(below_mask)

    The OAP stores information computed from the
    :func:`shaw_opening_angle_method`. See the two properties of the OAP
    :obj:`below_mask` and :obj:`sea_angles`.

    .. plot::
        :context:

        fig, ax = plt.subplots(1, 3, figsize=(10, 4))
        golfcube.quick_show('eta', idx=-1, ax=ax[0])
        im1 = ax[1].imshow(OAP.below_mask,
                           cmap='Greys_r')
        im2 = ax[2].imshow(OAP.sea_angles,
                           cmap='jet')
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

        # set variable info display options
        self._sea_angles_varinfo = plot.VariableInfo(
            'sea_angles', cmap=plt.cm.jet, label='opening angle')
        self._below_mask_varinfo = plot.VariableInfo(
            'below_mask', cmap=plt.cm.gray, label='where below')
        self._default_varinfo = self._sea_angles_varinfo

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

            # now check the type and allocate the arrays as xr.DataArray
            if isinstance(_below_mask, xr.core.dataarray.DataArray):
                self._below_mask = xr.zeros_like(_below_mask, dtype=bool)
                self._below_mask.name = 'below_mask'
                self._sea_angles = xr.zeros_like(_below_mask, dtype=float)
                self._sea_angles.name = 'sea_angles'
            elif isinstance(_below_mask, np.ndarray):
                # this will use meshgrid to fill out with dx=1 in shape of array
                self._below_mask = xr.DataArray(
                    data=np.zeros(_below_mask.shape, dtype=bool),
                    name='below_mask')
                self._sea_angles = xr.DataArray(
                    data=np.zeros(_below_mask.shape, dtype=float),
                    name='sea_angles')
            else:
                raise TypeError('Invalid type {0}'.format(type(_below_mask)))

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
        self._sea_angles[:] = sea_angles

        # properly assign the oceanmap to the self.below_mask
        #   set it to be bool regardless of input type
        self._below_mask[:] = below_mask.astype(bool)

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
        """Alias to `sea_angles`.

        This is the array that a contour is extracted from using some threshold
        value when making land and shoreline masks.
        """
        return self._sea_angles

    @property
    def data(self):
        return self._sea_angles


class MorphologicalPlanform(SpecialtyPlanform):
    """Planform for handling the morphological method.

    This `Planform` (called `MP` for short) is a wrapper/handler for the input
    and output from the :func:`morphological_closing_method`. The `MP` is a
    convenient way to manage extraction of a shoreline or a delta topset area.

    Moreoever, the `MP` can be used as the input for :doc:`many types of Mask
    </reference/mask/index>` objects, so it is often computationally
    advantageous to compute this `Planform` once, and then use it to create
    many different types of masks.

    The `MP` provides an alternative approach to shoreline and topset area
    delineation to the `OAM` method. Depending on the input parameters chosen,
    this method can be faster than the `OAM` method, however unlike the `OAM`
    method, the accuracy and quality of the extracted planform is sensitive to
    the parameter values and scales inherent in the supplied inputs.

    .. note::

        It is recommended to try several parameters using a sample slice of
        data before computing the `MP` for an entire dataset, as choice of
        input parameters will affect the speed and quality of results!

    .. plot::
        :context: reset
        :include-source:

        >>> golfcube = dm.sample_data.golf()
        >>> EM = dm.mask.ElevationMask(
        ...     golfcube['eta'][-1, :, :],
        ...     elevation_threshold=0)

        >>> MP = dm.plan.MorphologicalPlanform(EM, 10)

    The MP stores information computed from the
    :func:`morphological_closing_method`. See the property of the MP,
    the computed :obj:`mean_image` below.

    .. plot::
        :context:

        fig, ax = plt.subplots(1, 2, figsize=(7.5, 4))
        golfcube.quick_show('eta', idx=-1, ax=ax[0])
        im1 = ax[1].imshow(MP.mean_image,
                           cmap='cividis')
        dm.plot.append_colorbar(im1, ax=ax[1])
        ax[0].set_title('input elevation data')
        ax[1].set_title('MP.mean_image')
        for i in range(1, 2):
            ax[i].set_xticks([])
            ax[i].set_yticks([])

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
            ...     max_disk=3)
        """
        # make a temporary mask
        _em = mask.ElevationMask(
            elevation_data, **kwargs)

        # compute from __init__ pathway
        return MorphologicalPlanform(_em, max_disk, **kwargs)

    @staticmethod
    def from_mask(UnknownMask, max_disk, **kwargs):
        """Static method for creating a MorphologicalPlanform from a mask."""
        return MorphologicalPlanform(UnknownMask, max_disk, **kwargs)

    def __init__(self, *args, **kwargs):
        """Initialize the MorphologicalPlanform (MP).

        Initializing the MP requires at least a binary input mask representing
        the elevation or land area of the system. A secondary input setting
        the maximum disk size for morphological operations can be provided.

        .. warning::

            At this time two arguments are needed! Connections between the
            planform object and the cube are not yet implemented.

        Parameters
        ----------
        *args
            The first argument is expected to be an elevation mask, or an
            array which represents an elevation mask or land area. The
            expectation is that this input is the binary representation of
            the area from which you wish to identify the MP.

            The second argument is the maximum disk size for morphological
            operations in pixels. If a cube is connected and this argument is
            not supplied, the inlet width will be pulled from the cube's
            metadata and used to set this parameter.

        **kwargs
            Current supported key-word argument is 'allow_empty' which is a
            boolean argument that if True, allows the MP to be initialized with
            no other arguments supplied.

        .. note::

            Supplying elevation data or nonbinary data in general as the first
            argument to the MP will **not** result in an error, however the
            array will be coerced to be binary when morphological operations
            are performed. Therefore results when inputting non-binary data
            may not be what you expect.

        """
        super().__init__('morphological method', *args)
        self._shape = None
        self._elevation_mask = None
        self._max_disk = None

        # set variable info display options
        self._mean_image_varinfo = plot.VariableInfo(
            'mean_image', label='mean image')
        self._default_varinfo = self._mean_image_varinfo

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
        if issubclass(type(args[0]), mask.BaseMask):
            self._elevation_mask = args[0]._mask
        elif utils.is_ndarray_or_xarray(args[0]):
            self._elevation_mask = args[0]
        else:
            raise TypeError(
                'Type of first argument is unrecognized or unsupported')
        # now check the type and allocate the arrays as xr.DataArray
        if isinstance(self._elevation_mask, xr.core.dataarray.DataArray):
            self._mean_image = xr.zeros_like(self._elevation_mask, dtype=float)
            self._mean_image.name = 'mean_image'
        elif isinstance(self._elevation_mask, np.ndarray):
            # this will use meshgrid to fill out with dx=1 in shape of array
            self._mean_image = xr.DataArray(
                data=np.zeros(self._elevation_mask.shape, dtype=float),
                name='mean_image')
        else:
            raise TypeError(
                'Invalid type {0}'.format(type(self._elevation_mask)))

        # see if the inlet width is provided, if not see if cube is avail
        if (len(args) > 1):
            if isinstance(args[1], (int, float)):
                self._max_disk = int(args[1])
            else:
                raise TypeError(
                    'Expected single number to set max inlet size, got: '
                    '{0}'.format(args[1]))
        elif isinstance(self.cube, cube.BaseCube):
            try:
                self._max_disk = self.cube.meta['N0'].data
            except Exception:
                raise TypeError(
                    'Data cube does not contain metadata, you must '
                    'specify the inlet size.')
        else:
            raise TypeError(
                'Something went wrong. Check second input argument for '
                'inlet width.')

        self._shape = self._elevation_mask.shape

        # run the computation
        all_images, mean_image = morphological_closing_method(
            self._elevation_mask, biggestdisk=self._max_disk)

        # assign arrays to object
        self._mean_image[:] = np.ones_like(mean_image) - mean_image
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
        """Alias for `mean_image`.

        This is the array that a contour is extracted from using some threshold
        value when making land and shoreline masks.
        """
        return self._mean_image

    @property
    def data(self):
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
        golf.quick_show('eta', idx=15, ax=ax[0])
        ax[0].set_title('roughness = {:.2f}'.format(rgh0))
        golf.quick_show('eta', idx=-1, ax=ax[1])
        ax[1].set_title('roughness = {:.2f}'.format(rgh1))
        plt.show()
    """
    # extract data from masks
    if isinstance(land_mask, mask.LandMask):
        land_mask = land_mask.mask
        _lm = land_mask.values
        _dx = float(land_mask[land_mask.dims[0]][1] -
                    land_mask[land_mask.dims[0]][0])
    elif isinstance(land_mask, xr.core.dataarray.DataArray):
        _lm = land_mask.values
        _dx = float(land_mask[land_mask.dims[0]][1] -
                    land_mask[land_mask.dims[0]][0])
    elif isinstance(land_mask, np.ndarray):
        _lm = land_mask
        _dx = 1
    else:
        raise TypeError('Invalid type {0}'.format(type(land_mask)))

    _ = kwargs.pop('return_line', None)  # trash this variable if passed
    shorelength = compute_shoreline_length(
        shore_mask, return_line=False, **kwargs)

    # compute the length of the shoreline and area of land
    shore_len_pix = shorelength
    land_area_pix = np.sum(_lm) * _dx * _dx

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
        golf.quick_show('eta', idx=15, ax=ax[0])
        ax[0].set_title('length = {:.2f}'.format(len0))
        golf.quick_show('eta', idx=-1, ax=ax[1])
        ax[1].plot(line1[:, 0], line1[:, 1], 'r-')
        ax[1].set_title('length = {:.2f}'.format(len1))
        plt.show()
    """
    # check if mask or already array
    if isinstance(shore_mask, mask.ShorelineMask):
        shore_mask = shore_mask.mask
        _sm = shore_mask.values
        _dx = float(shore_mask[shore_mask.dims[0]][1] -
                    shore_mask[shore_mask.dims[0]][0])
    elif isinstance(shore_mask, xr.core.dataarray.DataArray):
        _sm = shore_mask.values
        _dx = float(shore_mask[shore_mask.dims[0]][1] -
                    shore_mask[shore_mask.dims[0]][0])
    elif isinstance(shore_mask, np.ndarray):
        _sm = shore_mask
        _dx = 1
        # should we have a warning that no dx was found here?
    else:
        raise TypeError('Invalid type {0}'.format(type(shore_mask)))

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
    dists_pts = np.sqrt((_x[~hit_pts]-_x[_closest])**2 +
                        (_y[~hit_pts]-_y[_closest])**2)
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
        dists_pts = np.sqrt((_x[~hit_pts]-line_xs_0[0])**2 +
                            (_y[~hit_pts]-line_ys_0[0])**2)
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
            dists_pts = np.sqrt((_x[~hit_pts]-_xi)**2 +
                                (_y[~hit_pts]-_yi)**2)
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

    # combine the xs and ys AND multiply by dx
    line = np.column_stack((line_xs, line_ys)) * _dx
    length = np.sum(np.sqrt((line_xs[1:]-line_xs[:-1])**2 +
                            (line_ys[1:]-line_ys[:-1])**2)) * _dx

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

    .. plot::
        :include-source:

        golf = dm.sample_data.golf()

        sm = dm.mask.ShorelineMask(
            golf['eta'][-1, :, :],
            elevation_threshold=0,
            elevation_offset=-0.5)

        # compute mean and stddev distance
        mean, stddev = dm.plan.compute_shoreline_distance(
            sm, origin=[golf.meta['CTR'].data, golf.meta['L0'].data])

        # make the plot
        fig, ax = plt.subplots()
        golf.quick_show('eta', idx=-1, ticks=True, ax=ax)
        ax.set_title('mean = {:.2f}'.format(mean))
        plt.show()
    """
    # check if mask or already array
    if isinstance(shore_mask, mask.ShorelineMask):
        shore_mask = shore_mask.mask
        _sm = shore_mask.values
        _dx = float(shore_mask[shore_mask.dims[0]][1] -
                    shore_mask[shore_mask.dims[0]][0])
    elif isinstance(shore_mask, xr.core.dataarray.DataArray):
        _sm = shore_mask.values
        _dx = float(shore_mask[shore_mask.dims[0]][1] -
                    shore_mask[shore_mask.dims[0]][0])
    elif isinstance(shore_mask, np.ndarray):
        _sm = shore_mask
        _dx = 1
    else:
        raise TypeError('Invalid type {0}'.format(type(shore_mask)))

    if not (np.sum(_sm) > 0):
        raise ValueError('No pixels in shoreline mask.')

    if _sm.ndim == 3:
        _sm = _sm.squeeze()

    # find where the mask is True (all x-y pairs along shore)
    _y, _x = np.argwhere(_sm).T

    # determine the distances (multiply by dx)
    _dists = np.sqrt((_x - origin[0])**2 + (_y - origin[1])**2) * _dx

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
    _iter = 0  # count number of closings, cap at 100
    while (_changed != 0) and (_iter < 100):
        _iter += 1
        _newimg = morphology.binary_closing(img, footprint=disk)
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
        is passed, the `.idx_trace` attribute will be used to query the
        `ChannelMask` and determine widths. Otherwise, an `Nx2` array can be
        passed, which specified the `dim1-dim2` coordinate pairs to use as the
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
        sec = dm.section.CircularSection(golf, radius_idx=40)

        # compute the metric
        m, s, w = dm.plan.compute_channel_width(
            cm, section=sec, return_widths=True)

        fig, ax = plt.subplots()
        cm.show(ax=ax, ticks=True)
        sec.show_trace('r-', ax=ax)
        ax.set_title(f'mean: {m:.2f}; stddev: {s:.2f}')
        plt.show()
    """
    if not (section is None):
        if issubclass(type(section), dm_section.BaseSection):
            section_trace = section.idx_trace
            section_coord = section._s.data
        elif isinstance(section, np.ndarray):
            section_trace = section
            section_coord = np.arange(len(section))
    else:
        # create one by default based on the channelmask?
        raise NotImplementedError()

    # check that the section trace is a valid shape
    #   todo...

    # coerce the channel mask to just the raw mask values
    if utils.is_ndarray_or_xarray(channelmask):
        if isinstance(channelmask, xr.core.dataarray.DataArray):
            _dx = float(channelmask[channelmask.dims[0]][1] -
                        channelmask[channelmask.dims[0]][0])
        elif isinstance(channelmask, np.ndarray):
            _dx = 1
    elif isinstance(channelmask, mask.ChannelMask):
        channelmask = channelmask.mask
        _dx = float(channelmask[channelmask.dims[0]][1] -
                    channelmask[channelmask.dims[0]][0])
        channelmask = np.array(channelmask)
    else:
        raise TypeError(
            'Input for `channelmask` was wrong type: {}.'.format(
                type(channelmask)))

    # get channel starts and ends
    _channelstarts, _channelends = \
        _get_channel_starts_and_ends(channelmask, section_trace)

    # compute the metric
    #   Note: channel widths are pulled from the coordinates of the section,
    #   which incorporate grid-spacing information. So, we DO NOT multiply
    #   the width by dx here.
    _channelwidths = (section_coord[_channelends - 1] -
                      section_coord[_channelstarts - 1])

    _m, _s = np.nanmean(_channelwidths), np.nanstd(_channelwidths)

    if return_widths:
        return _m, _s, _channelwidths
    else:
        return _m, _s


def _get_channel_starts_and_ends(channelmask, section_trace):
    """Get channel start and end coordinates (internal function).

    This function is used when calculating both channel widths and depths to
    get the start and end locations for channels along a given section trace.
    These indices are returned to the width/depth functions to ensure the
    computation of channel properties along a given section trace takes place
    over pixels identified as channel pixels (per the channel mask) only.

    .. important::

        section_trace must be the index coordinates of the section trace, and
        not the coordinate values that are returned from `section.idx_trace`.

    """
    _channelseries = channelmask[section_trace[:, 0],
                                 section_trace[:, 1]].astype(int)
    _padchannelseries = np.pad(_channelseries, (1,), 'constant',
                               constant_values=(False)).astype(int)
    _channelseries_diff = _padchannelseries[1:] - _padchannelseries[:-1]
    _channelstarts = np.where(_channelseries_diff == 1)[0]
    _channelstarts = np.where(_channelstarts == 0, 1, _channelstarts)
    _channelends = np.where(_channelseries_diff == -1)[0]
    return _channelstarts, _channelends


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
        is passed, the `.idx_trace` attribute will be used to query the
        `ChannelMask` and determine depths. Otherwise, an `Nx2` array can be
        passed, which specified the `dim1-dim2` coordinate pairs to use as the
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
            section_trace = section.idx_trace
            section_coord = section._s.data
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

    # get channel starts and ends
    _channelstarts, _channelends = \
        _get_channel_starts_and_ends(channelmask, section_trace)

    # compute channel widths
    _channelwidths = section_coord[_channelends-1] - section_coord[_channelstarts-1]

    # get the depth array along the section
    _depthslice = np.copy(depth)
    _depthseries = _depthslice[section_trace[:, 0], section_trace[:, 1]]

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


def compute_surface_deposit_time(data, surface_idx=-1, **kwargs):
    """Compute the time of last deposition for a single time.

    This method computes the timing of last deposition for each location for a
    given time_index. This is done by finding the last time the bed elevation
    was outside of a "stasis tolerance" length scale at each location.
    Therefore, the function requires the spacetime dataset of bed elevation,
    at a minimum, up to the point in time of interest.

    .. warning:: only works for indices (not times) as implemented.

    Parameters
    ----------
    data : :obj:`DataCube`, :obj:`ndarray`, :obj:`DataArray`
        Input data to compute the surface deposit time from. Must be `DataCube`
        or a array-like (`ndarray` or `DataArray`) containing bed elevation
        history, at a minimum, up until the time of interest.

    surface_idx : :obj:`int`
        The index along the time dimension of the array (`dim0`) to compute
        the surface deposit time for. This number cannot be greater than
        `data.shape[0]`.

    **kwargs
        Keyword arguments passed to supporting
        function :obj:`_compute_surface_deposit_time_from_etas`. Hint: you
        may want to control the `stasis_tol` parameter.

    Returns
    -------
    sfc_time : :obj:`ndarray`
        The time of the surface.

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset

        golf = dm.sample_data.golf()
        sfc_time = dm.plan.compute_surface_deposit_time(golf, surface_idx=-1)

        fig, ax = plt.subplots()
        im = ax.imshow(sfc_time)
        dm.plot.append_colorbar(im, ax=ax)
        plt.show()

    The keyword argument `stasis_tol` is an important control on the resultant
    calculation.

    .. plot::
        :include-source:
        :context: close-figs

        fig, ax = plt.subplots(1, 3, figsize=(10, 3))
        for i, tol in enumerate([1e-16, 0.01, 0.1]):
            i_sfc_date = dm.plan.compute_surface_deposit_time(
                golf, surface_idx=-1, stasis_tol=tol)
            im = ax[i].imshow(i_sfc_date)
            plt.colorbar(im, ax=ax[i], shrink=0.4)
            ax[i].set_title(f'stasis_tol={tol}')
        plt.show()

    """
    # sanitize the input surface declared
    if surface_idx == 0:
        raise ValueError(
            '`surface_idx` must not be 0 '
            ' (i.e., this would yield no timeseries)')

    if isinstance(data, cube.DataCube):
        etas = data['eta'][:surface_idx, :, :]
        etas = np.array(etas)  # strip xarray for input to helper
    elif utils.is_ndarray_or_xarray(data):
        etas = np.array(etas[:surface_idx, :, :])
    else:
        # implement other options...
        raise TypeError(
            'Unexpected data type input: {0}'.format(type(data)))

    sfc_date = _compute_surface_deposit_time_from_etas(etas, **kwargs)

    return sfc_date


def compute_surface_deposit_age(data, surface_idx, **kwargs):
    """Compute the age (i.e., how much time ago) the surface was deposited.

    .. warning:: only works for indices (not times) as implemented.

    .. hint::

        This function internally uses :obj:`compute_surface_deposit_time` and
        is simply the current time/idx of interest minute the date of
        deposition (with handling for wrapping negative indices).

    Parameters
    ----------
    data : :obj:`DataCube`, :obj:`ndarray`, :obj:`DataArray`
        Input data to compute the surface deposit age from. Must be `DataCube`
        or a array-like (`ndarray` or `DataArray`) containing bed elevation
        history, at a minimum, up until the time of interest.

    surface_idx : :obj:`int`
        The index along the time dimension of the array (`dim0`) to compute
        the surface deposit age for. This number cannot be greater than
        `data.shape[0]`.

    **kwargs
        Keyword arguments passed to supporting
        function :obj:`_compute_surface_deposit_time_from_etas`. Hint: you
        may want to control the `stasis_tol` parameter.

    Examples
    --------

    .. plot::
        :include-source:

        golf = dm.sample_data.golf()
        sfc_time = dm.plan.compute_surface_deposit_age(golf, surface_idx=-1)

        fig, ax = plt.subplots()
        ax.imshow(sfc_time, cmap='YlGn_r')
        plt.show()
    """
    sfc_date = compute_surface_deposit_time(data, surface_idx, **kwargs)
    # handle indices less than 0
    if surface_idx < 0:
        # wrap to find the index
        surface_idx = surface_idx % data.shape[0]
    return surface_idx - sfc_date


def _compute_surface_deposit_time_from_etas(etas, stasis_tol=0.01):
    """Helper for surface deposit time/age calculations.

    Parameters
    ----------
    etas
        An array-like with bed elevation information.

    stasis_tol
        The length scale above which any change in bed elevation is considered
        signficant. Changes in bed elevation less than this threshold are
        considered to be stasis and ignored in the computation of surface
        deposit times. Must be nonzero and positive (``>0``).

    .. hint:

        :obj:`stasis_tol` can be passed as a keyword argument to the public
        functions :obj:`compute_surface_deposit_time`
        and :obj:`compute_surface_deposit_age` to control the threshold.

    Returns
    -------
    sfc_date : obj:`ndarray`
    """
    if not (stasis_tol > 0):
        raise ValueError(
            f'`stasis_tol must be nonzero and positive, but was {stasis_tol}')

    etaf = np.array(etas[-1, :, :])

    whr = np.abs(etaf - etas) < stasis_tol
    sfc_date = np.argmax(whr, axis=0)

    return sfc_date
