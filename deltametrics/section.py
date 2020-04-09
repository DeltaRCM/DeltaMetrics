import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



class BaseSection(object):
    """Base section object.

    Defines common attributes and methods of a section object.

    This object should wrap around many of the functions available from
    :obj:`~deltametrics.strat`.

    """
    def __init__(self, pts, attribute=None, limit=None):
        """
        Extract values from Cube at pts.

        Parameters
        ----------

        pts : ndarray
            two column ndarray defining the x-y coordinates to extract the section.

        attribute : str, list, optional
            Which attributes to extract from Cube at pts. If attribute is
            None, then get all of the availabe values in the cube.

        limit : list, ndarray
            Vertical limits to extract the section over. Use None or np.nan to
            specify the lowermost or uppermost lines. Default is to extract
            full section.

        """
        pass



class DipSection(object):
    """Base section object.

    Defines common attributes and methods of a section object.

    """
    def __init__(self, apex, angle):
        pass



class RadialSection(object):
    """Base section object.

    Defines common attributes and methods of a section object.

    """
    def __init__(self, apex, radius):
        pass
