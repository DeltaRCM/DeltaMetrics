__all__ = ['Cube']

import numpy as np
import matplotlib.pyplot as plt

class Cube(object):
    """Data cube object.

    Data cube object that contains the x-y-z(-t) information. It may have any
    number of attached attributes (grain size, mud frac, elevation). Time is
    assumed to be constant, unless specified otherwise.

    .. note::
        This object is going to be core to the success of the project.
    
    """
    def __init__(self, *args):
        """Initialize the Cube.
        """

        pass
