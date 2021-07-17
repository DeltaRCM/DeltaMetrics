Introduction to Planform Objects
================================

Multiple definitions of the delta planform are supported.
These planform objects can be used as starting points for binary mask computation.
If not specified explicitly, they are often implicitly part of binary mask creation.

Below we demonstrate instantiation of both the :obj:`~deltametrics.plan.OpeningAnglePlanform` and the :obj:`~deltametrics.plan.MorphologicalPlanform` objects.

The OpeningAnglePlanform
------------------------

This planform object is based around the Opening Angle Method [1]_.

.. todo::

  Add example here

.. [1] Shaw, J. B., Wolinsky, M. A., Paola, C., & Voller, V. R. (2008). An image‐based method for shoreline mapping on complex coasts. Geophysical Research Letters, 35(12).

The MorphologicalPlanform
-------------------------

This planform object uses mathematical morphology to identify the delta planform, inspired by Geleynse et al (2012) [2]_.

.. todo::

  Add example here

.. [2] Geleynse, N., Voller, V. R., Paola, C., & Ganti, V. (2012). Characterization of river delta shorelines. Geophysical research letters, 39(17).
