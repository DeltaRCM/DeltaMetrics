import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



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
        ax.imshow(self.data, cmap=cmap, **kwargs)
        plt.show()



class ChannelMask(BaseMask):
    """Channel mask.

    A channel mask object, helps enforce valid masking of channels.

    Examples
    --------
    
    Initialize the channel mask
        >>> cmsk = dm.ChannelMask(arr)

    And visualize the mask:
        >>> cmsk.show_mask()

    .. plot:: pyplots/mask/channelmask.py

    """
    def __init__(self, arr, is_mask=False):
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



class LandMask(BaseMask):
    """land-water mask.

    A land-water mask object, helps enforce valid masking of land water interface.

    Examples
    --------
    >>> lmsk = dm.LandMask(arr)

    """
    def __init__(self, arr):
        """Initialize the LandMask.

        Intializing the land-water mask requires an array of data, should be
        two-dimensional. 
        """
        super().__init__(mask_type='land-water', data=arr)

        self.other_atts = 10


    @property
    def property_for_just_land(self):
        """Who knows!
        """
        return self._property_for_just_land *3


    def a_land_function(self):
        """This is a wrapper to the below function.
        """
        return a_land_function(self.data)
