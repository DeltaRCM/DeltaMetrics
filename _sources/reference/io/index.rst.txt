.. api.observation:

***********************************
Input/output file operations
***********************************

The package specifies several data input/output utilities. Typically, these utilities will not be interacted with directly, but rather are used by the :obj:`~deltametrics.cube.Cube` or :doc:`../section/index` to handle slicing. The IO utilities are data agnostic, that is to say, they do not care about what the data *are*, but instead just connect by variable names. Datasets commonly used are  could be lidar scans, overhead photos, grain-size maps (DeltaRCM), or flow velocity (DeltaRCM).

By default, nothing is loaded from disk and into memory, and instead all slicing operations are handled on the fly.


The functions are defined in ``deltametrics.io``.


.. io functions
.. ===================

.. currentmodule:: deltametrics.io

.. autosummary::
	:toctree: ../../_autosummary

	BaseIO
	NetCDFIO
