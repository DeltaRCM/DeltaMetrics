golfcube = dm.sample_data.golf()
golfcube.register_section(
    'radial', dm.section.RadialSection(azimuth=45))

# show the location and the "velocity" variable
fig, ax = plt.subplots(2, 1, figsize=(8, 4))
golfcube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
golfcube.sections['radial'].show_trace('r--', ax=ax[0])
golfcube.sections['radial'].show('velocity', ax=ax[1])
plt.show()
