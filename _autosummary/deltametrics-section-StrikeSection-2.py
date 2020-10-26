rcm8cube = dm.sample_data.cube.rcm8()
sc8cube = dm.cube.StratigraphyCube.from_DataCube(rcm8cube)
sc8cube.register_section('strike_half', dm.section.StrikeSection(y=20, x=[0, 120]))
# >>>
# show the location and the "velocity" variable
fig, ax = plt.subplots(2, 1, figsize=(8, 4))
rcm8cube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
sc8cube.sections['strike_half'].show_trace('r--', ax=ax[0])
sc8cube.sections['strike_half'].show('velocity', ax=ax[1])
plt.show()
