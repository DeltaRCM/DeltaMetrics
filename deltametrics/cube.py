import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

from . import io
from . import plan
from . import section
from . import utils


class _DataDict(dict):

    def __init__(self, *args, **kwargs):
        dict.__init__(*args, **kwargs)

    def __repr__(self):
        _repr = [key + ': ' + str(val.shape) for key, val
                 in zip(self.keys(), self.values())]
        return '{' + str(', '.join(_repr))


class Cube(object):
    """Data cube object.

    Data cube object that contains t-x-y information. It may have any
    number of attached attributes (grain size, mud frac, elevation).

    .. note::
        `Cube` stores all data in memory by default. Optionally, the Cube can
        be configured to use datafile slicing oeprations on indexing. This
        flexibility supports larger file sizes, but it is slower.


    """

    def __init__(self, data, read=[]):
        """Initialize the Cube.

        Parameters
        ----------
        data : str, dict
            If data is type `str`, the string points to a NetCDF or HDF5 file
            that can be read. Typically this is used to directly import files
            output from the pyDeltaRCM model. Alternatively, pass a
            :obj:`dict` with keys indicating variable names, and values with
            corresponding t-x-y `ndarray` of data.

        read : bool, optional
            Which variables to read from dataset into memory. Special option
            for ``read=True`` to read all available variables into memory.

        """
        # self.data = data
        self.data = _DataDict({})
        if type(data) is str:
            self._connect_to_file(data_path=data)
            self._read_meta_from_file()
            self._connect_vars_to_file(self.variables)
        elif type(data) is dict:
            # handle a dict, arrays set up already, make an io class to wrap it
            raise NotImplementedError
        else:
            raise TypeError('Invalid type for "data": %s' % type(data))

        self._plan_list = []
        self._section_list = []

        self.colorset = utils.Colorset()

    def __getitem__(self, var):
        """Return the variable.

        Overload slicing operations for io.
        """
        if var in self._variables:
            return self.dataio[var]
        else:
            raise ValueError('No attribute of {cube} named {var}'.format(
                             cube=str(self), var=var))

    def _connect_to_file(self, data_path):
        """Deprecated"""
        _, ext = os.path.splitext(data_path)
        if ext == '.nc':
            self._dataio = io.NetCDF_IO(data_path)
        elif ext == '.hf5':  # ?????
            self._dataio = io.HDF_IO(data_path)
        else:
            raise ValueError(
                'Invalid file extension for "data_path": %s' % data_path)

    def _read_meta_from_file(self):
        self._variables = self._dataio.keys

    def _connect_vars_to_file(self, var_list=[]):
        # self._data = self._dataio  # does not create a new copy?
        for var in var_list:
            # self._dataio.
            pass

    def read(self, read):
        """Read variable into memory

        Parameters
        ----------
        read : :obj:`list` of :obj:`str`, `str`, `True`
            Which variables to read into memory. If ``True``, all available
            variables are read into memory.
        """
        if read is True:  # special case, read all variables
            read = self._dataio.variables
        elif type(read) is str:
            read = [read]
        else:
            raise TypeError('Invalid type for "read": %s ' % read)

        for var in read:
            self._dataio.read(var)

    @property
    def dataio(self):
        return self._dataio

    @property
    def variables(self):
        return self._variables

    @property
    def plan_list(self):
        return self._plan_list

    @property
    def plans(self):
        return self._plan_list

    def register_plan(self, name, PlanInstance):
        if not issubclass(type(PlanInstance), plan.BasePlan):
            raise TypeError
        if not type(name) is str:
            raise TypeError
        self._plan_list[name] = PlanInstance

    @property
    def section_list(self):
        return self._section_list

    @property
    def sections(self):
        return self._section_list

    def register_section(self, name, SectionInstance):
        if not issubclass(type(SectionInstance), section.BaseSection):
            raise TypeError
        if not type(name) is str:
            raise TypeError
        self._section_list[name] = SectionInstance

    def show_cube(self, var, t=-1, x=-1, y=-1, ax=None):
        raise NotImplementedError

    def show_plan(self, var, t=-1, ax=None, ticks=False):
        _plan = self[var][t]  # REPLACE WITH OBJECT RETURNED FROM PLAN

        if not ax:
            ax = plt.gca()

        ax.imshow(_plan, cmap=self.colorset[var])
        ax.set_xlabel('')
        # ax.set_title(self.colorset[var])

    def show_section(self, var, section_type='strike', section_args=[0]):
        raise NotImplementedError
