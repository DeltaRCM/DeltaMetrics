
.. rubric:: Mask lexicon and use

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
