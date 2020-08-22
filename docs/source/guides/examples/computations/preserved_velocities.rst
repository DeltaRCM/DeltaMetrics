Compute ordinariness of preserved strata
----------------------------------------

Several studies have discussed the "ordinariness" of stratigraphy [1]_ [2]_, which describes the observation that most fluvial outcrop exposures seem to record relatively normal river conditions, rather than extreme events. 

This assertion of "ordinariness" has been examined in the context of vertical excursions of bed elevation (whereby more extreme excursions represent extreme events; [2]_).
We can go further than this with the pyDeltaRCM model outputs, which co-locate bed elevation change with velocity, in essence recording the actual magnitude of a flow event rather than bed-elevation change.
With this paired information we can interrogate the velocity of deposits in the stratigraphic record.

.. plot:: examples/preserved_velocities.py
    :include-source:

**This is not intented to be a formal scientific rebuttal to the references listed above! It's just a simple demonstration of how to use DeltaMetrics to interrogate the stratigraphic record.**


.. rubric:: References

.. [1] Paola, C., Ganti, V., Mohrig, D., Runkel, A.C. and Straub, K.M. (2018) Time not our time: Physical controls on the preservation and measurement of geologic time. Annual Review of Earth and Planetary Sciences., 46, annurev-earth-082517-010129

.. [2] Ganti, V., Hajek, E. A., Leary, K., Straub, K. M., & Paola, C. (2020). Morphodynamic hierarchy and the fabric of the sedimentary record. Geophysical Research Letters, 47, e2020GL087921. doi: 10.1029/2020GL087921.
