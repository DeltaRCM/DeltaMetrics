golfcube = dm.sample_data.golf()
golfcube.register_section('demo', dm.section.StrikeSection(idx=10))

fig, ax = plt.subplots(figsize=(5, 3))
golfcube.show_plan('eta', t=-1, ax=ax, ticks=True)
ax.plot(golfcube.sections['demo'].trace[:, 0],
        golfcube.sections['demo'].trace[:, 1], 'r--')
