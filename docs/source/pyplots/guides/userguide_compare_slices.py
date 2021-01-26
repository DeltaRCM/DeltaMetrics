import matplotlib.gridspec as gs

rcm8cube = dm.sample_data.rcm8()
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))
sc8cube = dm.cube.StratigraphyCube.from_DataCube(rcm8cube, dz=0.05)
sc8cube.register_section('demo', dm.section.StrikeSection(y=10))

fig, ax = plt.subplots(1, 2, figsize=(8, 2))
rcm8cube.sections['demo'].show('velocity', ax=ax[0])
sc8cube.sections['demo'].show('velocity', ax=ax[1])
