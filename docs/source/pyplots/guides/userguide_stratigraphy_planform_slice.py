golfcube = dm.sample_data.golf()

stratcube = dm.cube.StratigraphyCube.from_DataCube(golfcube, dz=0.05)
stratcube.register_section('demo', dm.section.StrikeSection(distance_idx=10))

elev_idx = (np.abs(stratcube.z - -2)).argmin()  # find nearest idx to -2 m

fig, ax = plt.subplots(figsize=(5, 3))
stratcube.show_plan('sandfrac', elev_idx, ticks=True)
