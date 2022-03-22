Radial growth of delta through time
-----------------------------------

The growth pattern of a delta can be predicted from the volume of a cylinder. 
The expectation is that a delta's planform area grows according to a power law that is a function of basin depth, sediment input, and time.

.. math::

    r = \sqrt{\frac{2 t Q_s}{h_b \pi}}

.. note::

    This formulation ignores deposit porosity (as is true for the sample data from `pyDeltaRCM` below).


.. plot::
    :include-source:
    :context: reset

    golf = dm.sample_data.golf()

    # measure the delta shoreline distance at five differet times
    time_idxs = np.linspace(15, golf.shape[0]-1, num=5, dtype=int)

    shoredist_mean = np.zeros(time_idxs.shape)
    shoredist_std = np.zeros(time_idxs.shape)
    for i, time_idx in enumerate(time_idxs):
        # compute the shoreline mask
        SM_mpm = dm.mask.ShorelineMask(
            golf['eta'][time_idx, :, :],
            elevation_threshold=0,
            method='MPM',
            contour_threshold=0.75,
            max_disk=8)
        SM_mpm.trim_mask(length=3)
        
        # compute the mean shoreline distance
        shoredist_mean[i], shoredist_std[i] = dm.plan.compute_shoreline_distance(
            SM_mpm, origin=(golf.meta['CTR'].data, golf.meta['L0'].data))

Now plot

.. plot::
    :include-source:
    :context:

    # make a predictive model
    def predict_for_t(t, Qs, hb):
        """Predict the delta shoreline radius.
        """
        return np.sqrt((2*t*Qs) / (hb * np.pi))

    # set up the parameters
    hb = golf.meta['hb'].data  # basin depth, m
    Qs = (golf.meta['h0'].data *
          golf.meta['u0'][0].data *
          golf.meta['N0'].data *
          golf.meta['dx'].data *
          (golf.meta['C0_percent'][0].data / 100))  # sediment input, m3/s
    t = np.linspace(0, golf.t[time_idxs[-1]], num=100)

    # make the figure
    fig, ax = plt.subplots()

    ax.plot(
        t, predict_for_t(t, Qs, hb))
    ax.errorbar(
        golf.t[time_idxs], shoredist_mean, shoredist_std,
        c='r', ls='none')
    ax.plot(
        golf.t[time_idxs], shoredist_mean,
        c='r', marker='o')

    ax.set_ylabel('radius (m)')
    ax.set_xlabel('time (s)')
    plt.show()

Why do the data not line up with the prediction?
A likely possibility is that the shoreline location determined from the :obj:`~deltametrics.plan.MorphologicalPlanform` does not include the full area of the deposit.

.. hint::
    
    Try passing the `elevation_offset` parameter to the MPM method, which is then passed along to the :obj:`~deltametrics.mask.ElevationMask`. 
