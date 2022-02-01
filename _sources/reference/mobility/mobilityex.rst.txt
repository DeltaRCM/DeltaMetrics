
Using the Mobility Functions
----------------------------

To use the mobility functions, you need a set of masks covering various time
points from the model output. These can be in the form of a list of mask
objects, numpy ndarrays, or xarrays. Or these can be 3-D arrays or xarrays.

For this example a few masks will be generated and put into lists. Then these
lists of masks will be used to compute metrics of channel mobility,
which will then be plotted.

.. note:: needs to be expanded!

.. plot::
    :include-source:
    :context: reset

    >>> golfcube = dm.sample_data.golf()
    >>> channelmask_list = []
    >>> landmask_list = []

    >>> for i in range(50, 60):
    ...     landmask_list.append(dm.mask.LandMask(
    ...         golfcube['eta'][i, ...], elevation_threshold=0))
    ...     channelmask_list.append(dm.mask.ChannelMask(
    ...         golfcube['eta'][i, ...], golfcube['velocity'][i, ...],
    ...         elevation_threshold=0, flow_threshold=0.3))

    >>> dryfrac = dm.mobility.calculate_channel_decay(
    ...     channelmask_list, landmask_list,
    ...     basevalues_idx=[0, 1, 2], window_idx=5)
    >>> Ophi = dm.mobility.calculate_planform_overlap(
    ...     channelmask_list, landmask_list,
    ...     basevalues_idx=[0, 1, 2], window_idx=5)
    >>> fr = dm.mobility.calculate_reworking_fraction(
    ...     channelmask_list, landmask_list,
    ...     basevalues_idx=[0, 1, 2], window_idx=5)
    >>> PwetA = dm.mobility.calculate_channel_abandonment(
    ...     channelmask_list,
    ...     basevalues_idx=[0, 1, 2], window_idx=5)

    >>> fig, ax = plt.subplots(2, 2)
    >>> dryfrac.plot.line(x='time', ax=ax[0, 0])
    >>> ax[0, 0].set_title('Dry Fraction')
    >>> Ophi.plot.line(x='time', ax=ax[0, 1])
    >>> ax[0, 1].set_title('Overlap Values')
    >>> fr.plot.line(x='time', ax=ax[1, 0])
    >>> ax[1, 0].set_title('Reworked Fraction')
    >>> PwetA.plot.line(x='time', ax=ax[1, 1])
    >>> ax[1, 1].set_title('Abandoned Fraction')
    >>> plt.tight_layout()
    >>> plt.show()
