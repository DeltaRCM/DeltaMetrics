golfcube = dm.sample_data.golf()
OAP = dm.plan.OpeningAnglePlanform.from_elevation_data(
  golfcube['eta'][-1, :, :], elevation_threshold=0)