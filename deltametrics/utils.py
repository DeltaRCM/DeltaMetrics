import os, sys
import abc

import netCDF4
import numpy as np
import matplotlib


def _get_version():
    """
    Extract version number from single file, and make it availabe everywhere.
    """
    from . import _version
    return _version.__version__()

def known_variables():
    return ['x', 'y', 'time', 'eta', 'stage', 'depth', 'discharge',
            'velocity', 'strata_age', 'strata_sand_frac', 'strata_depth']


class Base_IO(abc.ABC):
    """BaseIO object other file format wrappers inheririt from.

    .. note::
        This is an abstract class and cannot be instantiated directly. If you
        wish to subclass to create a new IO format, you must implement the
        following methods: ``load``, .
    """
    def __init__(self, data_path):
        """Initialize the base IO.
        """
        self.known_variables = known_variables()
        self.data_path = data_path

        self.connect()


    @property
    def data_path(self):
        """`str` : Path to data file.

        Parameters
        ----------
        data_path : str
            path to data file for IO operations.
        """
        return self._data_path
    

    @data_path.setter
    def data_path(self, var):
        self._data_path = var


    @abc.abstractmethod
    def connect(self):
        """Should connect to the data file.

        This function should initialize the file if it does not exist, or
        connect to the file if it already exists---but *do not* load the file.
        """
        return


    @abc.abstractmethod
    def load(self):
        """Should load data into memory.
        """
        return


    @abc.abstractmethod
    def write(self):
        """Should write the data to file.

        Take a :obj:`~deltametrics.cube.Cube` and write it to file.
        """
        return


    @abc.abstractmethod
    def __getitem__(self):
        """Should slice the data from file.
        """
        return



class NetCDF_IO(Base_IO):
    """Utility for consistent IO with netCDF files.

    This module wraps calls to the netCDF4 python module in a consistent API,
    so the user can work seamlessly with either netCDF4 files or HDF5 files.
    The public methods of this class are consistent with
    :obj:`~deltametrics.utils.HDF_IO`.
    """

    def __init__(self, data_path, load, write=False):
        """Initialize the NetCDF_IO handler.

        Parameters
        ----------
        data_path : str
            Path to file to read or write to.

        write : bool, optional
            Whether to allow writing to an existing file. Set to False by
            default, if a file already exists at ``data_path``, writing is
            disabled, unless ``write`` is set to True.

        """
        super().__init__(data_path=data_path)
        self.write = write
        self.type = 'netcdf'



    def connect(self):
        """Connect to the data file.

        Initialize the file if it does not exist, or simply ``return`` if the
        file already exists.

        .. note::
            This function is automatically called during initialization of a
            :obj:`~deltametrics.cube.Cube`.
        """
        if not os.path.isfile(self.data_path):
            self.dataset = netCDF4.Dataset(self.data_path, "w", format="NETCDF4")
        else:
            if self.write:
                self.dataset = netCDF4.Dataset(self.data_path, "r+")
            else:
                self.dataset = netCDF4.Dataset(self.data_path, "r")


    def load(self):
        """Convert `variables` in netCDF file to `ndarray`.

        This operation is usually used in preparation for coersion into a
        :obj:`~deltametrics.cube.Cube` instance.
        """
        _data = {}
        for var in self.dataset.variables:
            _arr = self.dataset.variables[var]
            _data[var] = np.array(_arr)

        # REMOVE TIME!
        return _data

    """
    NEED A DEFINITION FOR KEYS IF FROM FILE???


    """
    # def _convert_to_ndarray(self):
        
        
    def write(self):
        """Should write the data to file.

        Take a :obj:`~deltametrics.cube.Cube` and write it to file.
        """
        raise NotImplementedError
    
    def __getitem__(self):
        pass



class HDF_IO(Base_IO):
    """Utility for consistent IO with HDF5 files.

    This module wraps calls to the hdf5 python module in a consistent API,
    so the user can work seamlessly with either HDF5 files or netCDF4 files.
    The public methods of this class are consistent with
    :obj:`~deltametrics.utils.NetCDF_IO`.
    """

    def __init__(self, data_path):
        """Initialize the HDF5_IO handler

        parameters
        """
        raise NotImplementedError



class Colorset(object):
    """A default set of colors for attributes.

    This makes it easy to have consistent plots.
    """
    class _ColorsetDecorators(object):
        @classmethod
        def colormap(self, func=None):
            """Decorator to register colormaps.
            """
            self.parent().variable_list.append(func.__name__)

    def __init__(self, override_list=None):
        """Initialize the Colorset.

        Initialize the set with default colormaps. 

        ..note::
            It is expected that any attribute added to known_list has a valid
            property defined in this class.

        Parameters
        ----------
        override : dict, optional
            Dictionary defining variable-colormap sets. Dictionary value must
            be either a string (and then match defined colormaps in matplotlib
            or DeltaMetrics), a new matplotlib Colormap object, or an Mx3
            numpy array that can be coerced into a linear colormap.
        """
        # self.known_variables = known_variables()
        # self.variable_list = self.known_variables.copy()

        # for var in self.known_list:
        #     self.map[var] = 
        #     # setattr(self, var, self.variable_list[var])

        # self.map = {}
        # # initialize to default values (set in each attr)
        # for var in self.known_list:
        #     self.map[var] = self[var]
        #     # setattr(self, var, self.variable_list[var])



        self.known_list = known_variables()

        # self.override_list = override_list
        # self.variable_list = {**self.known_list, **self.override_list}
        for var in self.known_list:
            setattr(self, var, None)
        if override_list:
            if not type(override_list) is dict:
                raise TypeError('Invalid type for "override_list".'
                                'Must be type dict, but was type: %s ' % \
                                type(self.override_list))
            for var in override_list:
                if var in self.known_list:
                    setattr(self, var, override_list[var])
                else:
                    setattr(self, var, override_list[var])
    """A FALLBACK:
                    self.map[var] = override_list[var]

    def __getattr__(self, var):
        return self.map[var]

    """

    def __getitem__(self, var):
        print(var)
        return getattr(self, var)

    @property
    def variable_list(self):
        return self._variable_list
    
    @variable_list.setter
    def variable_list(self, var):
        if type(var) is dict:
            self._variable_list = var
        else:
            raise TypeError('Invalid type for "override_list".'
                            'Must be type dict, but was type: %s ' % type(self.override_list))

    @property
    def net_to_gross(self):
        return _net_to_gross

    @net_to_gross.setter
    def net_to_gross(self):
        self._net_to_gross = matplotlib.cm.get_cmap('viridis', 512)


    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, var):
        if not var:
            self._x = matplotlib.cm.get_cmap('viridis', 512)
        else:
            self._x = var


    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, var):
        if not var:
            self._y = matplotlib.cm.get_cmap('viridis', 512)
        else:
            self._y = var


    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, var):
        if not var:
            self._time = matplotlib.cm.get_cmap('viridis', 512)
        else:
            self._time = var


    @property
    def eta(self):
        return self._eta

    @eta.setter
    def eta(self, var):
        if not var:
            self._eta = matplotlib.cm.get_cmap('cividis', 512)
        else:
            self._eta = var


    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self, var):
        if not var:
            self._stage = matplotlib.cm.get_cmap('viridis', 512)
        else:
            self._stage = var


    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, var):
        if not var:
            self._depth = matplotlib.cm.get_cmap('viridis', 512)
        else:
            self._depth = var


    @property
    def discharge(self):
        return self._discharge

    @discharge.setter
    def discharge(self, var):
        if not var:
            self._discharge = matplotlib.cm.get_cmap('winter', 512)
        else:
            self._discharge = var


    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, var):
        if not var:
            self._velocity = matplotlib.cm.get_cmap('plasma', 512)
        else:
            self._velocity = var


    @property
    def strata_age(self):
        return self._strata_age

    @strata_age.setter
    def strata_age(self, var):
        if not var:
            self._strata_age = matplotlib.cm.get_cmap('viridis', 512)
        else:
            self._strata_age = var


    @property
    def strata_sand_frac(self):
        return self._strata_sand_frac

    @strata_sand_frac.setter
    def strata_sand_frac(self, var):
        if not var:
            self._strata_sand_frac = matplotlib.cm.get_cmap('viridis', 512)
        else:
            self._strata_sand_frac = var


    @property
    def strata_depth(self):
        return self._strata_depth

    @strata_depth.setter
    def strata_depth(self, var):
        if not var:
            self._strata_depth = matplotlib.cm.get_cmap('viridis', 512)
        else:
            self._strata_depth = var



def format_number(number):
    integer = int(round(number, -1))
    string = "{:,}".format(integer)
    return(string)



def format_table(number):
    integer = (round(number, 1))
    string = str(integer)
    return(string)
