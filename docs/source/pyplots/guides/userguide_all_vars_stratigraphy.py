rcm8cube = dm.sample_data.rcm8()

sc8cube = dm.cube.StratigraphyCube.from_DataCube(rcm8cube, dz=0.05)
sc8cube.register_section('demo', dm.section.StrikeSection(y=10))

fig, ax = plt.subplots(7, 1, sharex=True, sharey=True, figsize=(12, 9))
ax = ax.flatten()
for i, var in enumerate(['time'] + sc8cube.dataio.known_variables):
    sc8cube.show_section('demo', var, ax=ax[i], label=True,
                         style='shaded', data='stratigraphy')
