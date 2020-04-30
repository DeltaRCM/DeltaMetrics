import os
import warnings
import time
import abc

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from . import io
from . import plan
from . import section
from . import utils
from . import plot


class CubeVariable(np.ndarray):
    """Slice of a Cube.

    Slicing an :obj:`~deltametrics.cube.Cube` returns an object of this type.
    The ``CubeVariable`` is essentially a thin wrapper around the numpy
    ``ndarray``, enabling additional iniformation to be augmented, a
    lighter-weight __repr__, and flexibility for development.

    .. warning::
        You probably should not instantiate objects of this type directly.

    Examples
    --------
    .. doctest::

        >>> import deltametrics as dm
        >>> rcm8cube = dm.sample_data.cube.rcm8()

        >>> type(rcm8cube['velocity'])
        <class 'deltametrics.cube.CubeVariable'>
        >>> type(rcm8cube['velocity'].base)
        <class 'numpy.ndarray'>
        >>> rcm8cube['velocity'].variable
        velocity
    """

    def __new__(cls, *args, **kwargs):
        """Initialize the ndarray.
        """
        variable = kwargs.pop('variable', None)
        obj = np.array(*args, **kwargs)
        obj = np.asarray(obj).view(cls)
        obj.variable = variable
        return obj

    def __array_finalize__(self, obj):
        """Place things that must always happen here.

        This method is called when any new array is cast. The ``__new__``
        method is not called when views are case of an existing
        `CubeVariable`, so anything special about the `CubeVariable` needs to
        implemented here additionally. This could be configured to return a
        standard ndarray if a view is cast, but currently, it will return
        another CubeVariable instance.
        """
        if obj is None:
            return
        self.variable = getattr(obj, 'variable', None)

    def __repr__(self):
        if self.size > 50:
            return str(type(self)) + ' variable: `' + self.variable \
                + '`; dtype: ' + str(self.dtype) + ' size: ' + str(self.size)
        else:
            return str(self)


class BaseCube(abc.ABC):
    """Base cube object.

    Cube objects contain t-x-y or z-x-y information.

    This base class should not be used directly, but is subclassed below,
    providing convenient obejcts for maniplating common data types.

    .. note::
        `Cube` does not load any data into memory by default. This means that
        slicing is handled "behind the scenes" by an :doc:`I/O file handler
        </reference/io/index>`. Optionally, you can load files into memory for
        (sometimes) faster operations.
        See the :meth:`read` for more information.

    """

    def __init__(self, data, read=[], varset=None):
        """Initialize the BaseCube.

        Parameters
        ----------
        data : :obj:`str`, :obj:`dict`
            If data is type `str`, the string points to a NetCDF or HDF5 file
            that can be read. Typically this is used to directly import files
            output from the pyDeltaRCM model. Alternatively, pass a
            :obj:`dict` with keys indicating variable names, and values with
            corresponding t-x-y `ndarray` of data.

        read : :obj:`bool`, optional
            Which variables to read from dataset into memory. Special option
            for ``read=True`` to read all available variables into memory.

        varset : :class:`~deltametrics.plot.VariableSet`, optional
            Pass a `~deltametrics.plot.VariableSet` instance if you wish
            to style this cube similarly to another cube.
        """

        if type(data) is str:
            self._data_path = data
            self._connect_to_file(data_path=data)
            self._read_meta_from_file()
        elif type(data) is dict:
            # handle a dict, arrays set up already, make an io class to wrap it
            self._data_path = None
            raise NotImplementedError
        else:
            raise TypeError('Invalid type for "data": %s' % type(data))

        self._plan_set = {}
        self._section_set = {}

        if varset:
            self.varset = varset
        else:
            self.varset = plot.VariableSet()

    def __getitem__(self, var):
        """Return the variable.

        Overload slicing operations for io to return a
        :obj:`~deltametrics.cube.CubeVariable` instance when slicing.

        Parameters
        ----------
        var : :obj:`str`
            Which variable to slice.

        Returns
        -------
        CubeVariable : `~deltametrics.cube.CubeVariable`
            The instantiated CubeVariable.
        """

        if var in self._variables:
            return CubeVariable(self.dataio[var], variable=var)
        elif var == 'time':
            # a special attribute we add, which matches eta.shape
            _eta = self.dataio[var]
            _t = np.expand_dims(np.arange(_eta.shape[0]), axis=1)
            return CubeVariable(np.tile(_t, (1, *_eta.shape[1:])),
                                varable='time')
        else:
            raise AttributeError('No variable of {cube} named {var}'.format(
                                 cube=str(self), var=var))

    def _connect_to_file(self, data_path):
        """Connect to file.

        This method is used internally to send the ``data_path`` to the
        correct IO handler.
        """
        _, ext = os.path.splitext(data_path)
        if ext == '.nc':
            self._dataio = io.NetCDFIO(data_path)
        elif ext == '.hf5':  # ?????
            self._dataio = io.HDFIO(data_path)
        else:
            raise ValueError(
                'Invalid file extension for "data_path": %s' % data_path)

    def _read_meta_from_file(self):
        """Read metadata information from variables in file.

        This method is used internally to gather some useful info for
        navigating the variable trees in the stored files.
        """
        self._variables = self._dataio.keys

    def read(self, variables):
        """Read variable into memory

        Parameters
        ----------
        variables : :obj:`list` of :obj:`str`, :obj:`str`
            Which variables to read into memory.
        """
        if variables is True:  # special case, read all variables
            variables = self._dataio.variables
        elif type(variables) is str:
            variables = [variables]
        else:
            raise TypeError('Invalid type for "variables": %s ' % variables)

        for var in variables:
            self._dataio.read(var)

    @property
    def varset(self):
        """:class:`~deltametrics.plot.VariableSet` : Variable styling for plotting.

        Can be set with :code:`cube.varset = VariableSetInstance` where
        ``VariableSetInstance`` is a valid instance of
        :class:`~deltametrics.plot.VariableSet`.
        """
        return self._varset

    @varset.setter
    def varset(self, var):
        if type(var) is plot.VariableSet:
            self._varset = var
        else:
            raise TypeError('Pass a valid VariableSet instance.')

    @property
    def data_path(self):
        """:obj:`str` : Path connected to for file IO.

        Returns a string if connected to file, or None if cube initialized from ``dict``.
        """
        return self._data_path

    @property
    def dataio(self):
        """:obj:`~deltametrics.io.BaseIO` subclass : Data I/O handler.
        """
        return self._dataio

    @property
    def variables(self):
        """`list` : List of variable names as strings.
        """
        return self._variables

    @property
    def plan_set(self):
        """`dict` : Set of plan instances.
        """
        return self._plan_set

    @property
    def plans(self):
        """`dict` : Set of plan instances.

        Alias to :meth:`plan_set`.
        """
        return self._plan_set

    def register_plan(self, name, PlanInstance):
        """Register a planform to the cube.
        """
        if not issubclass(type(PlanInstance), plan.BasePlan):
            raise TypeError
        if not type(name) is str:
            raise TypeError
        self._plan_set[name] = PlanInstance

    @property
    def section_set(self):
        """:obj:`dict` : Set of section instances.
        """
        return self._section_set

    @property
    def sections(self):
        """:obj:`dict` : Set of section instances.

        Alias to :meth:`section_set`.
        """
        return self._section_set

    def register_section(self, name, SectionInstance):
        """Register a section to the :meth:`section_set`.

        Connect a section to the cube.

        Parameters
        ----------
        name : :obj:`str`
            The name to register the section.

        SectionInstance : :obj:`~deltametrics.section.BaseSection` subclass instance
            The section instance that will be registered.
        """

        if not issubclass(type(SectionInstance), section.BaseSection):
            raise TypeError
        if not type(name) is str:
            raise TypeError
        SectionInstance.connect(self)  # attach cube
        self._section_set[name] = SectionInstance

    def show_cube(self, var, t=-1, x=-1, y=-1, ax=None):
        """Show the cube in a 3d axis.
        """
        raise NotImplementedError

    def show_plan(self, var, t=-1, ax=None, ticks=False):
        """Show planform image.

        .. warning::
            NEEDS TO BE PORTED OVER TO WRAP THE .show() METHOD OF PLAN!
        """

        _plan = self[var][t]  # REPLACE WITH OBJECT RETURNED FROM PLAN

        if not ax:
            ax = plt.gca()

        ax.imshow(_plan,
                  cmap=self.varset[var].cmap,
                  norm=self.varset[var].norm,
                  vmin=self.varset[var].vmin,
                  vmax=self.varset[var].vmax)
        ax.set_xlabel('')
        # ax.set_title(self.varset[var])

    def show_section(self, *args, **kwargs):
        """Show a section.

        Can be called by name if section is already registered, or pass a
        fresh section instance and it will be connected.

        Wraps the Section's :meth:`~deltametrics.section.BaseSection.show`
        method.
        """

        if len(args) == 0:
            raise ValueError
        elif len(args) == 1:
            SectionInstance = args[0]
            SectionAttribute = None
        elif len(args) == 2:
            SectionInstance = args[0]
            SectionAttribute = args[1]

        if type(SectionInstance) is str:
            self.sections[SectionInstance].show(SectionAttribute, **kwargs)
        else:
            if not issubclass(type(SectionInstance), section.BaseSection):
                raise TypeError
            SectionInstance.show(**kwargs)


class DataCube(BaseCube):
    """DataCube object.

    DataCube contains t-x-y information. It may have any
    number of attached attributes (grain size, mud frac, elevation).
    """

    def __init__(self, data, read=[], varset=None, stratigraphy_from=None):
        """Initialize the BaseCube.

        Parameters
        ----------
        data : :obj:`str`, :obj:`dict`
            If data is type `str`, the string points to a NetCDF or HDF5 file
            that can be read. Typically this is used to directly import files
            output from the pyDeltaRCM model. Alternatively, pass a
            :obj:`dict` with keys indicating variable names, and values with
            corresponding t-x-y `ndarray` of data.

        read : :obj:`bool`, optional
            Which variables to read from dataset into memory. Special option
            for ``read=True`` to read all available variables into memory.

        varset : :class:`~deltametrics.plot.VariableSet`, optional
            Pass a `~deltametrics.plot.VariableSet` instance if you wish
            to style this cube similarly to another cube. If no argument is
            supplied, a new default VariableSet instance is created.

        stratigraphy_from : :obj:`str`, optional
            Pass a string that matches a variable name in the dataset to
            compute preservation and stratigraphy using that variable as
            elevation data. Typically, this is ``'eta'`` in pyDeltaRCM model
            outputs. Stratigraphy can be computed on an existing data cube
            with the :meth:`~deltametrics.cube.DataCube.stratigraphy_from`
            method.
        """

        super().__init__(data, read, varset)

        self._knows_spacetime = True
        self._knows_stratigraphy = False

        if stratigraphy_from:
            self.stratigraphy_from(variable=stratigraphy_from)

    def stratigraphy_from(self, variable='eta'):
        """Compute stratigraphy attributes.

        .. note::

            This method wraps the private method ``_compute_stratigraphy``.

        Parameters
        ----------
        variable : :obj:`str`, optional
            Which variable to use as elevation data for computing
            preservation. If no value is given for this parameter, we try to
            find a variable `eta` and use that for elevation data if it
            exists.
        """

        self._compute_stratigraphy(variable)
        self._knows_stratigraphy = True

    def _compute_stratigraphy(self, variable):
        """Compute stratigraphy attributes.

        Compute what is preserved. We can precompute several attributes of the
        stratigraphy, including which voxels are preserved, what their row
        indicies in the sparse stratigraphy matrix are, and what the elevation
        of each elevation entry in the final stratigraphy are. *This allows
        placing of any t-x-y stored variable into the section.*

        Currently, we store many of these computations as private attributes
        of the Cube itself. This is fine for a small cube, but may become
        cumbersome or intractable for large datasets.
        problematic as the matrix becomes very large.

        .. note::
            This could be ported out to wrap around a generalized function for
            computing stratigraphic representations of cubes. E.g., wrapping
            some function in :obj:`~deltametrics.strat`.
        """

        # copy out _eta for faster access, not retained in memory (?)
        _eta = np.array(self[variable], copy=True)
        _psvd = np.zeros_like(_eta)  # boolean for if retained
        _strata = np.zeros_like(_eta)  # elevation of surface at each t

        nt = _strata.shape[0]
        _strata[-1, :, :] = _eta[-1, :, :]
        _psvd[-1, :, :] = True
        for j in np.arange(nt - 2, -1, -1):
            _strata[j, ...] = np.minimum(_eta[j, ...],
                                         _strata[j + 1, ...])
            _psvd[j, :, :] = np.less(_eta[j, ...],
                                     _strata[j + 1, ...])

        self._psvd_vxl_cnt = _psvd.sum(axis=0, dtype=np.int)
        self._psvd_vxl_idx = _psvd.cumsum(axis=0, dtype=np.int)
        self._psvd_vxl_cnt_max = int(self._psvd_vxl_cnt.max())
        self._psvd_idx = _psvd.astype(np.bool)  # guarantee bool

        # Determine the elevation of any voxel that is preserved.
        # These are matrices that are size n_preserved-x-y.
        #    psvd_vxl_eta : records eta for each entry in the preserved matrix.
        #    psvd_flld    : fills above with final eta entry (for pcolormesh).
        self._psvd_vxl_eta = np.full((self._psvd_vxl_cnt_max,
                                      *_eta.shape[1:]), np.nan)
        self._psvd_flld = np.full((self._psvd_vxl_cnt_max,
                                   *_eta.shape[1:]), np.nan)
        for i in np.arange(_eta.shape[1]):
            for j in np.arange(_eta.shape[2]):
                self._psvd_vxl_eta[0:self._psvd_vxl_cnt[i, j], i, j] = _eta[
                    self._psvd_idx[:, i, j], i, j].copy()
                self._psvd_flld[0:self._psvd_vxl_cnt[i, j], i, j] = _eta[
                    self._psvd_idx[:, i, j], i, j].copy()
                self._psvd_flld[self._psvd_vxl_cnt[
                    i, j]:, i, j] = self._psvd_flld[
                    self._psvd_vxl_cnt[i, j] - 1, i, j]

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


class StratigraphicCube(BaseCube):
    """StratigraphicCube object.

    A cube of precomputed stratigraphy. This is a z-x-y matrix defining
    variables at specific voxel locations.

    .. warning::

        This class has not been implemented.
    """

    def __init__(self, data, read=[], varset=None):
        """Initialize the StratigraphicCube.

        Parameters
        ----------
        data : :obj:`str`, :obj:`dict`
            If data is type `str`, the string points to a NetCDF or HDF5 file
            that can be read. Typically this is used to directly import files
            output from the pyDeltaRCM model. Alternatively, pass a
            :obj:`dict` with keys indicating variable names, and values with
            corresponding t-x-y `ndarray` of data.

        read : :obj:`bool`, optional
            Which variables to read from dataset into memory. Special option
            for ``read=True`` to read all available variables into memory.

        varset : :class:`~deltametrics.plot.VariableSet`, optional
            Pass a `~deltametrics.plot.VariableSet` instance if you wish
            to style this cube similarly to another cube. If no argument is
            supplied, a new default VariableSet instance is created.
        """

        raise NotImplementedError
        super().__init__(data, read, varset)

        self._knows_spacetime = False
        self._knows_stratigraphy = True
