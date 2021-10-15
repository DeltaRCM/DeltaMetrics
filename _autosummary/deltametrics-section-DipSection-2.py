rcm8cube = dm.sample_data.rcm8()
sc8cube = dm.cube.StratigraphyCube.from_DataCube(rcm8cube)
sc8cube.register_section(
    'dip_short', dm.section.DipSection(y=[0, 50]))

# show the location and the "velocity" variable
fig, ax = plt.subplots(2, 1, figsize=(8, 4))
rcm8cube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
sc8cube.sections['dip_short'].show_trace('r--', ax=ax[0])
sc8cube.sections['dip_short'].show('velocity', ax=ax[1])
plt.show()
