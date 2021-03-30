
Mask lexicon and use
--------------------

.. rubric:: Computing masks efficiently

If you are creating multiple masks for the same planform data (e.g., for the same time slice of data for the physical domain), it is computationally advantageous to create a single :obj:`~deltametrics.plan.OpeningAnglePlanform`, and use this `OAP` to create other masks.

For example:

.. plot::
    :include-source:
    :context: reset

    >>> golfcube = dm.sample_data.golf()

    >>> OAP = dm.plan.OpeningAnglePlanform.from_elevation_data(
    ...     golfcube['eta'][-1, :, :],
    ...     elevation_threshold=0)

    >>> lm = dm.mask.LandMask.from_OAP(OAP)
    >>> sm = dm.mask.ShorelineMask.from_OAP(OAP)

    >>> fig, ax = plt.subplots(2, 2)
    >>> golfcube.show_plan('eta', t=-1, ax=ax[0, 0])
    >>> ax[0, 1].imshow(OAP.sea_angles, vmax=180, cmap='jet')
    >>> lm.show(ax=ax[1, 0])
    >>> sm.show(ax=ax[1, 1])


.. note:: needs to be expanded!



.. rubric:: Trimming masks before input to metrics and functions

Sometimes it is helpful to :meth:`trim a mask <~deltametrics.mask.BaseMask.trim_mask>`, essentially replacing values with a different value, before using the mask for analysis or input to functions.

.. plot::

    >>> golfcube = dm.sample_data.golf()

    >>> m0 = dm.mask.LandMask(
    ...     golfcube['eta'][-1, :, :],
    ...     elevation_threshold=0)
    >>> m1 = dm.mask.LandMask(
    ...     golfcube['eta'][-1, :, :],
    ...     elevation_threshold=0)

    # trim one of the masks
    >>> m1.trim_mask(length=3)

    >>> fig, ax = plt.subplots(1, 2)
    >>> m0.show(ax=ax[0])
    >>> m1.show(ax=ax[1])
    >>> plt.show()
