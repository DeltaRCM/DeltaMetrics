Creating Sections into different types of objects
-------------------------------------------------

TODO...

.. plot::
    :context: close-figs
    :include-source:

    golfcube = dm.sample_data.golf()
    OAP = dm.plan.OpeningAnglePlanform.from_elevation_data(
      golfcube['eta'][-1, :, :], elevation_threshold=0)


.. plot::
    :context: close-figs
    :include-source:

    MP = dm.plan.MorphologicalPlanform.from_elevation_data(
      golfcube['eta'][-1, :, :], elevation_threshold=0, max_disk=8)


.. plot::
    :context:
    :include-source:

    SM_from_OAM = dm.mask.ShorelineMask.from_Planform(
      OAP, contour_threshold=75)

    SM_from_MPM = dm.mask.ShorelineMask.from_Planform(
      MP, contour_threshold=0.75)

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    SM_from_OAM.show(ax=ax[0])
    ax[0].set_title('Shoreline from OAM')

    SM_from_MPM.show(ax=ax[1])
    ax[1].set_title('Shoreline from MPM')

    diff_im = ax[2].imshow(
      SM_from_OAM.mask.astype(float) - SM_from_MPM.mask.astype(float),
      interpolation=None, cmap='bone')
    ax[2].set_title('OAM shoreline - MPM shoreline')
    dm.plot.append_colorbar(diff_im, ax=ax[2])

    plt.tight_layout()
    plt.show()
