import matplotlib.gridspec as gs

golfcube = dm.sample_data.golf()
golfcube.stratigraphy_from('eta')
golfcube.register_section('demo', dm.section.StrikeSection(distance_idx=10))

stratcube = dm.cube.StratigraphyCube.from_DataCube(golfcube, dz=0.05)
stratcube.register_section('demo', dm.section.StrikeSection(distance_idx=10))

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 6))
golfcube.sections['demo'].show('time', style='lines',
                               data='stratigraphy',
                               ax=ax[0], label=True)
stratcube.sections['demo'].show('time', ax=ax[1])
golfcube.sections['demo'].show('time', data='stratigraphy',
                               ax=ax[2])
