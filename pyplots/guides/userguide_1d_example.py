rcm8cube = dm.sample_data.cube.rcm8()

ets = rcm8cube['eta'][:, 25, 120]  # a "real" slice of the model

fig, ax = plt.subplots(figsize=(8, 4))
dm.plot.show_one_dimensional_trajectory_to_strata(ets, ax=ax, dz=0.25)
