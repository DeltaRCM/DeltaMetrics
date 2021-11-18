import matplotlib.gridspec as gs

golfcube = dm.sample_data.golf()
golfcube.register_section('demo', dm.section.StrikeSection(distance_idx=10))
stratcube = dm.cube.StratigraphyCube.from_DataCube(golfcube, dz=0.05)
stratcube.register_section('demo', dm.section.StrikeSection(distance_idx=10))

fig, ax = plt.subplots(1, 2, figsize=(8, 2))
golfcube.sections['demo'].show('velocity', ax=ax[0])
stratcube.sections['demo'].show('velocity', ax=ax[1])
