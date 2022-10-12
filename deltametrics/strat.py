import abc
import warnings

import numpy as np
import xarray as xr


def compute_trajectory():
    """Show 1d profile at point.
    """
    pass


def compute_compensation(line1, line2):
    """Compute compensation statistic betwen two lines.

    Explain the stat.

    Parameters
    ----------
    line1 : ndarray
        First surface to use (two-dimensional matrix with x-z coordinates of
        line).

    line2 : ndarray
        Second surface to use (two-dimensional matrix with x-z coordinates of
        line).

    Returns
    -------
    CV : float
        Compensation statistic.

    """
    pass


def compute_boxy_stratigraphy_volume(elev, prop, sigma_dist=None,
                                     z=None, dz=None, nz=None,
                                     return_cube=False):
    """Process t-x-y data volume to boxy stratigraphy volume.

    This function returns a "frozen" cube of stratigraphy
    with values of the supplied property/variable (:obj:`prop`) placed into a
    three-dimesional real-valued 3D array.

    By default, the data are returned as a numpy `ndarray`, however, specify
    :obj:`return_Cube` as `True` to return a
    :obj:`~deltametrics.cube.FrozenStratigraphyCube`. If `False`, function
    :additionally returns an `ndarray` of elevations corresponding to the
    :stratigraphy positions.

    Parameters
    ----------
    elev : :obj:`ndarray`
        The `t-x-y` ndarry of elevation data to determine stratigraphy.

    prop : :obj:`ndarray`
        The `t-x-y` ndarry of property data to process into the stratigraphy.

    sigma_dist : :obj:`ndarray`, :obj:`float`, :obj:`int`, optional
        Optional subsidence distance argument used to adjust the elevation
        data to account for subsidence when computing stratigraphy. See
        :obj:`_adjust_elevation_by_subsidence` for a complete description.

    z : :obj:`ndarray`, optional
        Vertical coordinates for stratigraphy, in meters. Optional, and
        mutually exclusive with :obj:`dz` and :obj:`nz`,
        see :obj:`_determine_strat_coordinates` for complete description.

    dz : :obj:`float`, optional
        Vertical resolution of stratigraphy, in meters. Optional, and mutually
        exclusive with :obj:`z` and :obj:`nz`,
        see :obj:`_determine_strat_coordinates` for complete description.

    nz : :obj:`int`, optional
        Number of intervals for vertical coordinates of stratigraphy.
        Optional, and mutually exclusive with :obj:`z` and :obj:`dz`,
        see :obj:`_determine_strat_coordinates` for complete description.

    return_cube : :obj:`boolean`, optional
        Whether to return the stratigraphy as a
        :obj:`~deltametrics.cube.FrozenStratigraphyCube` instance. Default is
        to return an `ndarray` and :obj:`elevations` `ndarray`.

        .. warning:: not implemented!

    Returns
    -------
    stratigraphy_cube : :obj:`~deltametrics.cube.StratigraphyCube`
        Not Implemented.

    stratigraphy : :obj:`ndarray`
        A `z-x-y` `ndarray` with data from `prop` placed into voxels to fill
        stratigraphy.

    elevations : :obj:`ndarray`
        A `z-x-y` `ndarray` with elevations for each voxel in the stratigraphy
        array.
    """
    # verify dimensions
    if elev.shape != prop.shape:
        raise ValueError('Mismatched input shapes "elev" and "prop".')
    if elev.ndim != 3:
        raise ValueError('Input arrays must be three-dimensional.')

    # compute preservation from low-level funcs
    if sigma_dist is not None:
        # convert elevations
        elev = _adjust_elevation_by_subsidence(elev, sigma_dist)
    strata, _ = _compute_elevation_to_preservation(elev)
    z = _determine_strat_coordinates(elev, z=z, dz=dz, nz=nz)
    strata_coords, data_coords = _compute_preservation_to_cube(strata, z=z)

    # copy data out and into the stratigraphy based on coordinates
    nx, ny = strata.shape[1:]
    stratigraphy = np.full((len(z), nx, ny), np.nan)  # preallocate nans
    _cut = prop.values[data_coords[:, 0], data_coords[:, 1],
                       data_coords[:, 2]]
    stratigraphy[strata_coords[:, 0],
                 strata_coords[:, 1],
                 strata_coords[:, 2]] = _cut

    elevations = np.tile(z, (ny, nx, 1)).T

    if return_cube:
        raise NotImplementedError
    else:
        return stratigraphy, elevations


def compute_boxy_stratigraphy_coordinates(elev, sigma_dist=None,
                                          z=None, dz=None, nz=None,
                                          return_cube=False,
                                          return_strata=False):
    """Process t-x-y data volume to boxy stratigraphy coordinates.

    This function computes the corresponding preservation of `t-x-y`
    coordinates in a dense 3D volume of stratigraphy, as `z-x-y` coordinates.
    This "mapping" is able to be computed only once, and used many times to
    synthesize a cube of preserved stratigraphy from an arbitrary `t-x-y`
    dataset.

    Parameters
    ----------
    elev : :obj:`ndarray`
        The `t-x-y` ndarry of elevation data to determine stratigraphy.

    sigma_dist : :obj:`ndarray`, :obj:`float`, :obj:`int`, optional
        Optional subsidence distance argument used to adjust the elevation
        data to account for subsidence when computing stratigraphy. See
        :obj:`_adjust_elevation_by_subsidence` for a complete description.

    z : :obj:`ndarray`, optional
        Vertical coordinates for stratigraphy, in meters. Optional, and
        mutually exclusive with :obj:`dz` and :obj:`nz`,
        see :obj:`_determine_strat_coordinates` for complete description.

    dz : :obj:`float`, optional
        Vertical resolution of stratigraphy, in meters. Optional, and mutually
        exclusive with :obj:`z` and :obj:`nz`,
        see :obj:`_determine_strat_coordinates` for complete description.

    nz : :obj:`int`, optional
        Number of intervals for vertical coordinates of stratigraphy.
        Optional, and mutually exclusive with :obj:`z` and :obj:`dz`,
        see :obj:`_determine_strat_coordinates` for complete description.

    return_cube : :obj:`boolean`, optional
        Whether to return the stratigraphy as a
        :obj:`~deltametrics.cube.FrozenStratigraphyCube` instance. Default is
        `False`, do not return a cube.

        .. warning:: not implemented!

    return_strata : :obj:`boolean`, optional
        Whether to return the stratigraphy elevations, as returned from
        internal computations in :obj:`_compute_elevation_to_preservation`.
        Default is `False`, do not return strata.

    Returns
    -------
    stratigraphy_cube : :obj:`~deltametrics.cube.StratigraphyCube`
        Not Implemented.

    strata_coords : :obj:`ndarray`
        An `N x 3` array of `z-x-y` coordinates where information is preserved
        in the boxy stratigraphy. Rows in `strat_coords` correspond with rows
        in `data_coords`. See :obj:`_compute_preservation_to_cube` for
        computation implementation.

    data_coords : :obj:`ndarray`
        An `N x 3` array of `t-x-y` coordinates where information is to be
        extracted from the data array. Rows in `data_coords` correspond with
        rows in `strat_coords`.  See :obj:`_compute_preservation_to_cube` for
        computation implementation.

    strata : :obj:`ndarray`
        A `t-x-y` `ndarry` of stratal surface elevations. Returned as third
        argument only if `return_strata=True`.
    """
    # compute preservation from low-level funcs
    if sigma_dist is not None:
        # adjust elevation by subsidence rates if present
        elev = _adjust_elevation_by_subsidence(elev, sigma_dist)
    strata, _ = _compute_elevation_to_preservation(elev)
    z = _determine_strat_coordinates(elev, z=z, dz=dz, nz=nz)
    strata_coords, data_coords = _compute_preservation_to_cube(strata, z=z)

    if return_cube:
        raise NotImplementedError
    elif return_strata:
        return strata_coords, data_coords, strata
    else:
        return strata_coords, data_coords


class BaseStratigraphyAttributes(object):

    def __init__(self, style):
        self._style = style

    @abc.abstractmethod
    def __call__(self):
        """Slicing operation to get sections and planforms.

        Must be implemented by subclasses.
        """
        ...

    def __getitem__(self, *unused_args, **unused_kwargs):
        raise NotImplementedError('Use "__call__" to slice.')

    @property
    def style(self):
        return self._style

    @property
    def display_arrays(self):
        return self.data, self.X, self.Y

    @property
    @abc.abstractmethod
    def data(self):
        return ...

    @property
    @abc.abstractmethod
    def X(self):
        return ...

    @property
    @abc.abstractmethod
    def Y(self):
        return ...

    @property
    @abc.abstractmethod
    def preserved_index(self):
        """:obj:`ndarray` : Boolean array indicating preservation.

        True where data is preserved in final stratigraphy.
        """
        return ...

    @property
    @abc.abstractmethod
    def preserved_voxel_count(self):
        """:obj:`ndarray` : Nmber of preserved voxels per x-y.

        X-Y array indicating number of preserved voxels per x-y pair.
        """
        return ...


class BoxyStratigraphyAttributes(object):
    """Attribute set for boxy stratigraphy information, emebdded into a DataCube.
    """

    def __init__(self):
        super().__init__('boxy')
        raise NotImplementedError(
            'Implementation should match MeshStratigraphyAttributes')


class MeshStratigraphyAttributes(BaseStratigraphyAttributes):
    """Attribute set for mesh stratigraphy information, emebdded into a DataCube.

    This object stores attributes of stratigraphy as a "mesh", only retaining
    the minimal necessary information to represent the stratigraphy. Contrast
    this with Boxy stratigraphy.

    Notes
    -----
    Some descriptions regarding implementation.

    _psvd_idx : :obj:`ndarray` of `bool`
        Preserved index into the section array.

    _psvd_flld : :obj:`ndarray`
        Elevation of preserved voxels, filled to vertical extent with the
        final elevation of the bed (i.e., to make it displayable by
        pcolormesh).

    _i : :obj:`ndarray`
        Row index for sparse matrix of preserved strata. I.e., which row
        in the stratigraphy matrix each voxel "winds up as".

    _j : :obj:`ndarray`
        Column index for sparse matrix of preserved strata. I.e., which
        column in the stratigraphy matrix each voxel "winds up as". This
        is kind of just a dummy to make the api consistent with ``_i``,
        because the column cannot change with preservation.
    """

    def __init__(self, elev, **kwargs):
        """
        We can precompute several attributes of the stratigraphy, including
        which voxels are preserved, what their row indicies in the sparse
        stratigraphy matrix are, and what the elevation of each elevation
        entry in the final stratigraphy are. *This allows placing of any t-x-y
        stored variable into the section.*

        Parameters
        ---------

        elev :
            elevation t-x-y array to compute from

        Keyword arguments
        -----------------
        load : :obj:`bool`, optional
            Whether to load the eta array into memory before computation. For
            especially large data files, `load` should be `False` to keep the
            file on disk; note on-disk computation is considerably slower.

        sigma_dist : :obj:`ndarray`, :obj:`float`, :obj:`int`, optional
            Optional subsidence distance argument that is used to adjust the
            elevation data to account for subsidence when computing
            stratigraphy. See :obj:`_adjust_elevation_by_subsidence` for a
            complete description.
        """
        super().__init__('mesh')

        # load or read eta field
        load = kwargs.pop('load', True)
        if load:
            _eta = np.array(elev)
        else:
            _eta = elev

        # rate of subsidence
        sigma_dist = kwargs.pop('sigma_dist', None)
        if sigma_dist is not None:
            _eta = _adjust_elevation_by_subsidence(_eta, sigma_dist)

        # make computation
        _strata, _psvd = _compute_elevation_to_preservation(_eta)
        _psvd[0, ...] = True
        self.strata = _strata

        self.psvd_vxl_cnt = _psvd.sum(axis=0, dtype=int)
        self.psvd_vxl_idx = _psvd.cumsum(axis=0, dtype=int)
        self.psvd_vxl_cnt_max = int(self.psvd_vxl_cnt.max())
        self.psvd_idx = _psvd.astype(bool)  # guarantee bool

        # Determine the elevation of any voxel that is preserved.
        # These are matrices that are size n_preserved-x-y.
        #    psvd_vxl_eta : records eta for each entry in the preserved matrix.
        #    psvd_flld    : fills above with final eta entry (for pcolormesh).
        self.psvd_vxl_eta = np.full((self.psvd_vxl_cnt_max,
                                     *_eta.shape[1:]), np.nan)
        self.psvd_flld = np.full((self.psvd_vxl_cnt_max,
                                  *_eta.shape[1:]), np.nan)
        for i in np.arange(_eta.shape[1]):
            for j in np.arange(_eta.shape[2]):
                self.psvd_vxl_eta[0:self.psvd_vxl_cnt[i, j], i, j] = _eta[
                    self.psvd_idx[:, i, j], i, j].copy()
                self.psvd_flld[0:self.psvd_vxl_cnt[i, j], i, j] = _eta[
                    self.psvd_idx[:, i, j], i, j].copy()
                self.psvd_flld[self.psvd_vxl_cnt[i, j]:, i, j] = self.psvd_flld[  # noqa: E501
                    self.psvd_vxl_cnt[i, j] - 1, i, j]

    def __call__(self, _dir, _x0, _x1):
        """Get a slice out of the stratigraphy attributes.

        Used for building section variables.

        Parameters
        ----------
        _dir : :obj:`str`
            Which direction to slice. If 'section', then _x0 is the
            _coordinates to slice in the domain length, and _x1 is the
            coordinates _to slice in the domain width direction.

        _x0, _x1

        Returns
        -------
        strat_attr : :obj:`dict`
            Dictionary containing useful information for sections and plans
            derived from the call.
        """
        strat_attr = {}
        if _dir == 'section':
            strat_attr['strata'] = self.strata[:, _x0, _x1]
            strat_attr['psvd_idx'] = _psvd_idx = self.psvd_idx[:, _x0, _x1]
            strat_attr['psvd_flld'] = self.psvd_flld[:, _x0, _x1]
            strat_attr['x0'] = _i = self.psvd_vxl_idx[:, _x0, _x1]
            strat_attr['x1'] = _j = np.tile(np.arange(_i.shape[1]),
                                            (_i.shape[0], 1))
            strat_attr['s'] = _j[0, :]          # along-sect coord
            strat_attr['s_sp'] = _j[_psvd_idx]  # along-sect coord, sparse
            strat_attr['z_sp'] = _i[_psvd_idx]  # vert coord, sparse

        elif (_dir == 'plan') or (_dir == 'planform'):
            pass
            # cannot be done without interpolation for mesh strata.
            # should be possible for boxy stratigraphy?
        else:
            raise ValueError('Bad "_dir" argument: %s' % str(_dir))
        return strat_attr

    @property
    def data(self):
        return self._data

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @property
    def preserved_index(self):
        """:obj:`ndarray` : Boolean array indicating preservation.

        True where data is preserved in final stratigraphy.
        """
        return self._psvd_idx

    @property
    def preserved_voxel_count(self):
        """:obj:`ndarray` : Nmber of preserved voxels per x-y.

        X-Y array indicating number of preserved voxels per x-y pair.
        """
        return self._psvd_vxl_cnt


def _compute_elevation_to_preservation(elev):
    """Compute the preserved elevations of stratigraphy.

    Given elevation data alone, we can compute the preserved stratal surfaces.
    These surfaces depend on the timeseries of bed elevation at any spatial
    location. We determine preservation by marching backward in time,
    determining when was the most recent time that the bed elevation was equal
    to a given elevation.

    This function is declared as private and not part of the public API,
    however some users may find it helpful. The function is heavily utlized
    internally. Function inputs and outputs are standard numpy `ndarray`, so
    that these functions can accept data from an arbitrary source.

    Parameters
    ----------
    elev : :obj:`ndarray` or :obj:`xr.core.dataarray.DataArray`
        The `t-x-y` volume of elevation data to determine stratigraphy.

    Returns
    -------
    strata : :obj:`ndarray`
        A `t-x-y` `ndarry` of stratal surface elevations.

    psvd : :obj:`ndarray`
        A `t-x-y` boolean `ndarry` of whether a `(t,x,y)` point of
        instantaneous time is preserved in any of the final stratal surfaces.
        To determine whether time from a given *timestep* is preserved, use
        ``psvd.nonzero()[0] - 1``.
    """
    psvd = np.zeros_like(elev.data, dtype=bool)  # bool, if retained
    strata = np.zeros_like(elev.data)  # elev of surface at each t

    nt = strata.shape[0]
    if isinstance(elev, np.ndarray) is True:
        _elev = elev
    elif isinstance(elev, xr.core.dataarray.DataArray) is True:
        _elev = elev.values
    else:  # case where elev is a CubeVariable
        _elev = elev.data.values

    strata[-1, ...] = _elev[-1, ...]
    for j in np.arange(nt - 2, -1, -1):
        strata[j, ...] = np.minimum(_elev[j, ...],
                                    strata[j + 1, ...])
        psvd[j + 1, ...] = np.less(strata[j, ...],
                                   strata[j + 1, ...])
    if nt > 1:  # allows a single-time elevation-series to return
        psvd[0, ...] = np.less(strata[0, ...],
                               strata[1, ...])

    return strata, psvd


def _compute_preservation_to_time_intervals(psvd):
    """Compute the preserved timesteps.

    The output from :obj:`_compute_elevation_to_preservation` records whether
    an instance of time, defined exactly at the data interval, is recorded in
    the stratigraphy (here, "recorded" does not include stasis). This differs
    from determining which *time-intervals* are preserved in the stratigraphy,
    because the deposits reflect the conditions *between* the save intervals.

    While this computation is simply an offset-by-one indexing (``psvd[1:,
    ...]``), the function is implemented explicitly and utilized internally
    for consistency.

    .. note::

        `True` in the preserved time-interval array does not necessarily
        indicate that an entire timestep was preserved, but rather that some
        portion of this time-interval (up to the entire interval) is recorded.

    Parameters
    ----------
    psvd : :obj:`ndarray`
        Boolean `ndarray` indicating the preservation of instances of time.
        Time is expected to be the 0th axis.

    Returns
    -------
    psvd_intervals : :obj:`ndarray`
        Boolean `ndarray` indicating the preservation of time-intervals,
        including partial intervals.
    """
    return psvd[1:, ...]


def _compute_preservation_to_cube(strata, z):
    """Compute the cube-data coordinates to strata coordinates.

    Given elevation preservation data (e.g., data from
    :obj:`~deltametrics.strat._compute_elevation_to_preservation`), compute
    the coordinate mapping from `t-x-y` data to `z-x-y` preserved
    stratigraphy.

    While stratigraphy is time-dependent, preservation at any spatial x-y
    location is independent of any other location. Thus, the computation is
    vectorized to sweep through all "stratigraphic columns" simultaneously.
    The operation works by beginning at the highest elevation of the
    stratigraphic volume, and sweeping down though all elevations with an
    `x-y` "plate". The plate indicates whether sediments are preserved below
    the current iteration elevation, at each x-y location.

    Once the iteration elevation is less than the strata surface at any x-y
    location, there will *always* be sediments preserved below it, at every
    elevation. We simply need to determine which time interval these sediments
    record. Then we store this time indicator into the sparse array.

    So, in the end, coordinates in resultant boxy stratigraphy are linked to
    `t-x-y` coordinates in the data source, by building a mapping that can be
    utilized repeatedly from a single stratigraphy computation.

    This function is declared as private and not part of the public API,
    however some users may find it helpful. The function is heavily utlized
    internally. Function inputs and outputs are standard numpy `ndarray`, so
    that these functions can accept data from an arbitrary source.

    Parameters
    ----------
    strata : :obj:`ndarray`
        A `t-x-y` `ndarry` of stratal surface elevations. Can be computed by
        :obj:`~deltametrics.strat._compute_elevation_to_preservation`.

    z :
        Vertical coordinates of stratigraphy. Note that `z` does not need to
        have regular intervals.

    Returns
    -------
    strat_coords : :obj:`ndarray`
        An `N x 3` array of `z-x-y` coordinates where information is preserved
        in the boxy stratigraphy. Rows in `strat_coords` correspond with
        rows in `data_coords`.

    data_coords : :obj:`ndarray`
        An `N x 3` array of `t-x-y` coordinates where information is to be
        extracted from the data array. Rows in `data_coords` correspond
        with rows in `strat_coords`.
    """
    # preallocate boxy arrays and helpers
    plate = np.atleast_1d(np.zeros(strata.shape[1:], dtype=int))
    strat_coords, data_coords = [], []  # preallocate sparse idx lists
    _zero = np.array([0])

    # the main loop through the elevations
    seek_elev = strata[-1, ...]  # the first seek is the last surface
    for k in np.arange(len(z) - 1, -1, -1):  # for every z, from the top
        e = z[k]  # which elevation for this iteration
        whr = e < seek_elev  # where elev is below strat surface
        t = np.maximum(_zero, (np.argmin(strata[:, ...] <= e, axis=0)))
        plate[whr] = int(1)  # track locations in the plate

        xy = plate.nonzero()
        ks = np.full((np.count_nonzero(plate)), k)  # might be faster way
        idxs = t[xy]  # must happen before incrementing counter

        strat_ks = np.column_stack((ks, *xy))
        data_idxs = np.column_stack((idxs, *xy))
        strat_coords.append(strat_ks)  # list of numpy arrays
        data_coords.append(data_idxs)

    strat_coords = np.vstack(strat_coords)  # to single (N x 3) array
    data_coords = np.vstack(data_coords)
    return strat_coords, data_coords


def _determine_strat_coordinates(elev, z=None, dz=None, nz=None):
    """Return a valid Z array for stratigraphy based on inputs.

    This helper function enables support for user specified `dz`, `nz`, or `z`
    in many functions. The logic for determining how to handle these mutually
    exclusive inputs is placed in this function, to ensure consistent behavior
    across all classes/methods internally.

    .. note::

        If none of the optional parameters is supplied, then a default value
        is used of `dz=0.1`.

    .. important::

        Precedence when multiple arguments are supplied is `z`, `dz`, `nz`.

    Parameters
    ----------
    elev : :obj:`ndarray`
        An up-to-three-dimensional array with timeseries of elevation values,
        where elevation is expected to along the zeroth axis.

    z : :obj:`ndarray`, optional
        Array of Z values to use for bounding intervals (i.e., points in `z`),
        returned unchanged if supplied.

    dz : :obj:`float`, optional
        Interval in created Z array. Z array is created as
        ``np.arange(np.min(elev), np.max(elev)+dz, step=dz)``.

    nz : :obj:`int`, optional
        Number of *intervals* in `z`, that is, the number of points in `z` is
        `nz+1`. Z array is created as ``np.linspace(np.min(elev), np.max
        (elev), num=nz, endpoint=True)``.
    """
    # if nothing is supplied
    if (dz is None) and (z is None) and (nz is None):
        warnings.warn(
            UserWarning('No specification for stratigraphy spacing '
                        'was supplied. Default is to use `dz=0.1`'))
        # set the default option when nothing is supplied
        dz = 0.1

    # set up an error message to use in a few places
    _valerr = ValueError('"dz" or "nz" cannot be zero or negative.')

    # process to find the option to set up z
    if not (z is None):
        if np.isscalar(z):
            raise ValueError('"z" must be a numpy array.')
        return z
    elif not (dz is None):
        if dz <= 0:
            raise _valerr
        max_dos = np.max(elev.data) + dz  # max depth of section, meters
        min_dos = np.min(elev.data)       # min dos, meters
        return np.arange(min_dos, max_dos, step=dz)
    elif not (nz is None):
        if nz <= 0:
            raise _valerr
        max_dos = np.max(elev.data)
        min_dos = np.min(elev.data)
        return np.linspace(min_dos, max_dos, num=nz+1, endpoint=True)
    else:
        raise RuntimeError('No coordinates determined. Check inputs.')


def _adjust_elevation_by_subsidence(elev, sigma_dist):
    """Adjust elevation array by subsidence rates.
    Given the elevation information and information about the distance
    subsided, the true sedimentation and erosion pattern can be unraveled.
    The function is written flexibly to handle subsidence distance information
    provided as either a single constant value, a constant value over a 2-D
    spatial pattern, a 1-D temporal vector when the distance changes but
    is applied to the entire domain, and finally a full t-x-y 3-D array with
    subsidence information for each x-y position in the domain at each
    instance in time.
    Critically, when temporal information is provided (either 1-D vector or
    full 3-D array) values are assumed to be cumulative subsidence distances
    at each index in time. Alternatively, when constant information is given
    (1-D float/int, or 2-D x-y array) it is assumed that the constant values
    reported represent the distance subsided per temporal index.
    This function is declared as a private function and is not part of the
    public API. Higher level functions make a call internally to adjust the
    elevation data based on subsidence rate information when the optional
    argument `sigma_dist` is provided.

    Parameters
    ----------
    elev : :obj:`ndarray` or :obj:`xr.core.dataarray.DataArray`
        The `t-x-y` volume of elevation data to determine stratigraphy.

    sigma_dist : :obj:`ndarray`, :obj:`xr.core.dataarray.DataArray`,
                 :obj:`float`, :obj:`int`
        The subsidence information. If a single float or int is provided,
        that value is assumed to apply across the entire domain for all times.
        If a 2-D array is provided, the distances are assumed for each
        x-y location and are applied across all times. If a 1-D vector is
        provided, it is assumed that each value applies to all locations in
        the domain for a given time. If a full 3-D array is provided, it is
        assumed to follow the same `t-x-y` convention as the elevation data.
        Positive distances indicate subsidence, negative distances are uplift.
        
    Returns
    -------
    elev_adjusted : :obj:`ndarray`
        A `t-x-y` `ndarray` of the same shape as the input `elev` array, but
        the values have all been adjusted by the subsidence information
        provided by the input `sigma_dist`.
    """
    # single value assumed to be constant displacement over all time
    if isinstance(sigma_dist, (int, float)):
        s_arr = np.ones_like(elev) * sigma_dist
        s_arr[0] = 0.0  # no subsidence at time 0
        # assume dist subsided each timestep
        s_arr = np.flip(np.cumsum(s_arr, axis=0), axis=0)
    # 1-D array renaming when topo is 1-D too
    elif len(sigma_dist.shape) == 1 and len(elev.shape) == 1:
        s_arr = sigma_dist
        # convert to base of preserved strat at each time based on subs
        s_arr = np.max(s_arr) - s_arr
    # 2-D array gets cast into the shape of the 3-d elevation array
    elif len(sigma_dist.shape) == 2 and len(elev.shape) == 3:
        s_arr = np.tile(sigma_dist, (elev.shape[0], 1, 1))
        s_arr[0, ...] = 0.0  # no subsidence at time 0
        s_arr = np.flip(np.cumsum(s_arr, axis=0), axis=0)  # sum up over time
    elif len(sigma_dist.shape) == 1 and len(sigma_dist) == elev.shape[0] and \
      len(elev.shape) == 3:
        # proper 1-D timeseries; flip and rename to s_arr
        s_arr = sigma_dist
        # convert to base of preserved strat at each time based on subs
        s_arr = np.max(s_arr) - s_arr
        # casting for a 1-D vector of sigma that matches elev time dimension
        s_arr = np.tile(
            sigma_dist.reshape(len(sigma_dist), 1, 1),
            (1, elev.shape[1], elev.shape[2]))
    else:
        # else shapes of arrays must be the same
        if np.shape(elev) != np.shape(sigma_dist):
            raise ValueError(
                'Shapes of input arrays elev and sigma_dist do not match.')
        else:
            # have to change name of input sigma_dist to s_arr
            s_arr = sigma_dist
            # convert to base of preserved strat at each time based on subs
            s_arr = np.max(s_arr, axis=0) - s_arr
    # adjust and return elevation
    elev_adjusted = np.zeros_like(elev).astype(float)  # init adjusted array
    # do elevation adjustment - first dimension assumed to be time
    elev_adjusted = elev - s_arr
    # return the subsidence-adjusted elevation values
    return elev_adjusted
