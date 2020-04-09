"""Sample data cube

A sample of a data cube (x,y,t) for examples, tests, etc.

Example
-------

A Data cube can be instantiated as, for example::

  >>> import deltametrics as dm
  >>> _tulane = dm.sample_data.cube.tdb12()

Available information on the data cubes is enumerated in the following
section.


Example data cubes
------------------------

:meth:`tdb12` : `ndarray`
  This data cube is from Tulane Delta Basin expt 12. 

:meth:`rcm1` : `ndarray`
  This data cube is from the DeltaRCM model. 

"""


import numpy as np

def tdb12():
  return np.array([[]])


def rcm1():
  return np.array([[]])
