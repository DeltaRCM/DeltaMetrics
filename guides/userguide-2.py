# use a new cube
maskcube = dm.sample_data.golf()

# create the masks from variables in the cube
land_mask = dm.mask.LandMask(
    maskcube['eta'][-1, :, :],
    elevation_threshold=0)

wet_mask = dm.mask.WetMask(
    maskcube['eta'][-1, :, :],
    elevation_threshold=0)

channel_mask = dm.mask.ChannelMask(
    maskcube['eta'][-1, :, :],
    maskcube['velocity'][-1, :, :],
    elevation_threshold=0,
    flow_threshold=0.3)

centerline_mask = dm.mask.CenterlineMask(
    maskcube['eta'][-1, :, :],
    maskcube['velocity'][-1, :, :],
    elevation_threshold=0,
    flow_threshold=0.3)

edge_mask = dm.mask.EdgeMask(
    maskcube['eta'][-1, :, :],
    elevation_threshold=0)

shore_mask = dm.mask.ShorelineMask(
    maskcube['eta'][-1, :, :],
    elevation_threshold=0)