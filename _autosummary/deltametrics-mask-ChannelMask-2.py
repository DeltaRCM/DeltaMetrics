golfcube = dm.sample_data.golf()

# Create the ElevationMask
emsk = dm.mask.ElevationMask(
    golfcube['eta'][-1, :, :],
    elevation_threshold=0)

# Create the FlowMask
fmsk = dm.mask.FlowMask(
    golfcube['velocity'][-1, :, :],
    flow_threshold=0.3)

# Make the ChannelMask from the ElevationMask and FlowMask
cmsk = dm.mask.ChannelMask.from_mask(
    emsk, fmsk)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
golfcube.show_plan('eta', t=-1, ax=ax[0])
cmsk.show(ax=ax[1])
plt.show()