rcm8cube = dm.sample_data.rcm8()
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))


fig, ax = plt.subplots(figsize=(5, 3))
rcm8cube.show_plan('eta', t=-1, ax=ax, ticks=True)
ax.plot(rcm8cube.sections['demo'].trace[:, 0],
        rcm8cube.sections['demo'].trace[:, 1], 'r--')
