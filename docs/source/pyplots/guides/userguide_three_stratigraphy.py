import matplotlib.gridspec as gs

rcm8cube = dm.sample_data.cube.rcm8()
rcm8cube.stratigraphy_from('eta')
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))

sc8cube = dm.cube.StratigraphyCube.from_DataCube(rcm8cube, dz=0.05)
sc8cube.register_section('demo', dm.section.StrikeSection(y=10))

fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 6))
rcm8cube.sections['demo'].show('time', style='lines',
                               data='stratigraphy',
                               ax=ax[0], label=True)
sc8cube.sections['demo'].show('time', ax=ax[1])
rcm8cube.sections['demo'].show('time', data='stratigraphy',
                               ax=ax[2])
