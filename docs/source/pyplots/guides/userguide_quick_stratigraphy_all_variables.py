rcm8cube = dm.sample_data.rcm8()
rcm8cube.stratigraphy_from('eta')
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))


fig, ax = plt.subplots(7, 1, sharex=True, sharey=True, figsize=(12, 12))
ax = ax.flatten()
for i, var in enumerate(['time'] + rcm8cube.dataio.known_variables):
    rcm8cube.show_section('demo', var, ax=ax[i], label=True,
                          style='shaded', data='stratigraphy')
