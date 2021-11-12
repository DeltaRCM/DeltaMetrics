golfcube = dm.sample_data.golf()
golfcube.register_section(
    'circular', dm.section.CircularSection(radius=30))

# show the location and the "velocity" variable
fig, ax = plt.subplots(2, 1, figsize=(8, 4))
golfcube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
golfcube.sections['circular'].show_trace('r--', ax=ax[0])
golfcube.sections['circular'].show('velocity', ax=ax[1])
plt.show()
