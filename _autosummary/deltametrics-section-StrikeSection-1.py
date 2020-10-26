rcm8cube = dm.sample_data.cube.rcm8()
rcm8cube.register_section('strike', dm.section.StrikeSection(y=10))
# >>>
# show the location and the "velocity" variable
fig, ax = plt.subplots(2, 1, figsize=(8, 4))
rcm8cube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
rcm8cube.sections['strike'].show_trace('r--', ax=ax[0])
rcm8cube.sections['strike'].show('velocity', ax=ax[1])
plt.show()
