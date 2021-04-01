golfcube = dm.sample_data.golf()
_EM = dm.mask.ElevationMask(
    golfcube['eta'][-1, :, :],
    elevation_threshold=0)

# # extract a mask of area below sea level as the
# #   inverse of the ElevationMask
_below_mask = ~(_EM.mask)

OAP = dm.plan.OpeningAnglePlanform(_below_mask)
