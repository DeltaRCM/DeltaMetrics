import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



def a_land_function(mask):
    """Compute a land-water function

    This function does blah blah...

    Parameters
    ----------
    mask : land-waterMask
        A land-water mask instance.

    Returns
    -------
    float
        something who knows what, maybe has math: :math:`\\alpha`.
    """
    pass



def compute_delta_radius(mask):
    """Compute the delta radius

    This function does blah blah...

    Parameters
    ----------
    mask : land-waterMask
        A land-water mask instance.

    Returns
    -------
    float
        something who knows what, maybe has math: :math:`\\alpha`.
    """
    pass



def a_channel_function(mask):
    """Compute a channel function

    This function does blah blah...

    Parameters
    ----------
    mask : ChannelMask
        A channel mask instance.

    Returns
    -------
    float
        something who knows what, maybe has math: :math:`\\alpha`.
    """
    pass



def compute_shoreline_angles(mask, param2=False):
    """Compute shoreline angle.
    
    Computes some stuff according to:
   
    .. math::
    
        \\theta = 7 \\frac{\\rho}{10-\\tau}

    where :math:`\\rho` is something, :math:`\\tau` is another. 

    Parameters
    ----------

    mask : :obj:`~deltametrics.land.LandMask`
        LandMask with shoreline to compute along.

    param2 : float, optional 
        Something else? It's assumed to be false.


    Returns
    -------

    vol_conc : float
        Volumetric concentration.


    Examples
    --------

    >>> dm.coastline.compute_angles(landmask, true)
    0.54

    """
    return 0.54
