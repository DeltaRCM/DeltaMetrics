golfcube = dm.sample_data.golf()
emsk = dm.mask.ElevationMask(
    golfcube['eta'][-1, :, :],
    elevation_threshold=0)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
golfcube.show_plan('eta', t=-1, ax=ax[0])
emsk.show(ax=ax[1])
plt.show()