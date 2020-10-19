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
  calc_chan_decay
  calc_planform_overlap
  calc_reworking_fraction
  calc_chan_abandonment
  mobility_curve_fit
