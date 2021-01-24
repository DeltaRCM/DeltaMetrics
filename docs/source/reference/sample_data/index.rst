.. api.sample_data:

*********************************
Sample data
*********************************

The package includes several sample data sets to show how to interact with the
API. You do not *need* to get your data into the same format as these data, but
doing so will likely make it simpler to use DeltaMetrics, and to get the most
benefit from the tools included here.

The sample data are defined in ``deltametrics.sample_data``. 

.. currentmodule:: deltametrics.sample_data

The sample data cubes can be accessed as, for example:

.. doctest::

    >>> import deltametrics as dm
    >>> rcm8cube = dm.sample_data.rcm8()

.. note::

    Data is handled by `pooch` and will be downloaded and cached on local
    computer as needed.


Available information on the data cubes is enumerated in the following
section.


Example data cubes
------------------

.. autofunction:: golf
.. autofunction:: rcm8
.. autofunction:: landsat


Paths to data files
"""""""""""""""""""

.. note::

	The file path to each sample data cube can be accessed by a call to
	`sample_data._get_xxxxxx_path()`  for the corresponding data set.

.. doctest::

	>>> dm.sample_data._get_rcm8_path()
	'.../pyDeltaRCM_Output_8.nc'
