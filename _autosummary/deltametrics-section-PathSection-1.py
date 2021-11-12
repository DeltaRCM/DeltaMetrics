golfcube = dm.sample_data.golf()
golfcube.register_section('path', dm.section.PathSection(
    path=np.array([[50, 3], [65, 17], [130, 10]])))
# >>>
# show the location and the "velocity" variable
fig, ax = plt.subplots(2, 1, figsize=(8, 4))
golfcube.show_plan('eta', t=-1, ax=ax[0], ticks=True)
golfcube.sections['path'].show_trace('r--', ax=ax[0])
golfcube.sections['path'].show('velocity', ax=ax[1])
plt.show()
