.. api.mobile:

***********************************
Channel Mobility Functions
***********************************

This set of functions is for computing channel mobility metrics.
Inputs to these functions are some sort of binary channel masks, and maps
defining the fluvial surface (land area) over which the metrics should be
computed. Functionality for fitting linear, harmonic, and exponential curves
to the results of these functions is provided as well.


The functions are defined in ``deltametrics.mobility``.


.. mobility functions
.. ===================

.. currentmodule:: deltametrics.mobility

.. autosummary::
  :toctree: ../../_autosummary

  check_inputs
  calculate_channel_decay
  calculate_planform_overlap
  calculate_reworking_fraction
  calculate_channel_abandonment
