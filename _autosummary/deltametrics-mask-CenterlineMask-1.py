golfcube = dm.sample_data.golf()
cntmsk = dm.mask.CenterlineMask(
    golfcube['eta'][-1, :, :],
    golfcube['velocity'][-1, :, :],
    elevation_threshold=0,
    flow_threshold=0.3)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
golfcube.show_plan('eta', t=-1, ax=ax[0])
cntmsk.show(ax=ax[1])
plt.show()