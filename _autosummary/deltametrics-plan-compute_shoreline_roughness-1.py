golf = dm.sample_data.golf()

# early in model run
lm0 = dm.mask.LandMask(
    golf['eta'][15, :, :],
    elevation_threshold=0,
    elevation_offset=-0.5)
sm0 = dm.mask.ShorelineMask(
    golf['eta'][15, :, :],
    elevation_threshold=0,
    elevation_offset=-0.5)

# late in model run
lm1 = dm.mask.LandMask(
    golf['eta'][-1, :, :],
    elevation_threshold=0,
    elevation_offset=-0.5)
sm1 = dm.mask.ShorelineMask(
    golf['eta'][-1, :, :],
    elevation_threshold=0,
    elevation_offset=-0.5)