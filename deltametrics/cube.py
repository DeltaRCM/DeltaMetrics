import os
import numpy as np
import matplotlib.pyplot as plt

from . import utils

class Cube(object):
    """Data cube object.

    Data cube object that contains t-x-y information. It may have any
    number of attached attributes (grain size, mud frac, elevation).

    .. note::
        `Cube` stores all data in memory by default. Optionally, the Cube can
        be configured to use datafile slicing oeprations on indexing. This
        flexibility supports larger file sizes, but it is slower.

    
    """
    def __init__(self, data, load=True):
        """Initialize the Cube.

        Parameters
        ----------
        data : str, dict
            If data is type `str`, the string points to a NetCDF or HDF5 file
            that can be read. Typically this is used to directly import files
            output from the pyDeltaRCM model. Alternatively, pass a
            :obj:`dict` with keys indicating variable names, and values with
            corresponding t-x-y `ndarray` of data.

        load : bool, optional
            Whether to load dataset into memory.

        .. warning::
            Option ``load = True`` is currently not implemented and will throw
            an error.
        """
        self.data = data, load
        self.colorset = utils.Colorset()

    @property
    def in_memory(self):
        """`bool` : Whether dataset is loaded into memory.
        """
        return self._in_memory
    

    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, var):
        try:
            data, load = var
        except ValueError:
            raise ValueError("Pass an iterable with two items")
        else:
            if type(data) is str:
                _, ext = os.path.splitext(data)
                if ext == '.nc':
                    io = utils.NetCDF_IO(data_path=data, load=load)
                elif ext == 'hf': # ?????
                    io = utils.HDF_IO(data_path=data, load=load)
                else:
                    raise ValueError('Invalid file extension for "data_path": %s' % data)
            elif type(data) is dict:
                raise NotImplementedError # handle a dict, arrays set up already
            else:
                raise TypeError('Invalid type for "data_path": %s' % type(data))
        
        if load:
            self._data = io.load()
            self._variables = self.data.keys()
            self._in_memory = True
        else:
            self._data = io
            self._in_memory = False

    @property
    def variables(self):
        return self._variables


    def __getattr__(self, var):
        """Return the variable.

        Overload slicing operations for io.
        """
        if var in self._data.keys():
            return self._data[var]
        else:
            raise ValueError
        # todo: implement error handling if not a value of _data either



    def imshow(self, attr=None, ax=None):
        pass