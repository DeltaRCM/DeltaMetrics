golfcube = dm.sample_data.golf()
golfcube.register_section('demo', dm.section.StrikeSection(distance_idx=10))

fig, ax = plt.subplots(figsize=(5, 3))
golfcube.quick_show('eta', idx=-1, ax=ax, ticks=True)
ax.plot(golfcube.sections['demo'].trace[:, 0],
        golfcube.sections['demo'].trace[:, 1], 'r--')
