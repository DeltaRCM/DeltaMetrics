golfcube = dm.sample_data.golf()
golfcube.register_section('strike', dm.section.StrikeSection(y=10))
# >>>
# show the location and the "velocity" variable
fig, ax = plt.subplots(2, 1, figsize=(8, 4))
golfcube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
golfcube.sections['strike'].show_trace('r--', ax=ax[0])
golfcube.sections['strike'].show('velocity', ax=ax[1])
plt.show()
