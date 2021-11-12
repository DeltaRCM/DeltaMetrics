golfcube = dm.sample_data.golf()

ets = golfcube['eta'].data[:, 10, 85]  # a "real" slice of the model

fig, ax = plt.subplots(figsize=(8, 4))
dm.plot.show_one_dimensional_trajectory_to_strata(ets, ax=ax, dz=0.25)
