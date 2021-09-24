MP = dm.plan.MorphologicalPlanform.from_elevation_data(
  golfcube['eta'][-1, :, :], elevation_threshold=0, max_disk=5)