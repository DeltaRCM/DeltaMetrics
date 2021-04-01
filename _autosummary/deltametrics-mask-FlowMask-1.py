golfcube = dm.sample_data.golf()
fvmsk = dm.mask.FlowMask(
    golfcube['velocity'][-1, :, :],
    flow_threshold=0.3)
fdmsk = dm.mask.FlowMask(
    golfcube['discharge'][-1, :, :],
    flow_threshold=4)

fig, ax = plt.subplots(1, 2)
fvmsk.show(ax=ax[0])
fdmsk.show(ax=ax[1])
plt.show()