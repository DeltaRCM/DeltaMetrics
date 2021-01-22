rcm8cube = dm.sample_data.rcm8()
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))

fig, ax = plt.subplots(1, 3, figsize=(6, 2))
rcm8cube.show_plan('eta', t=40, ax=ax[0])
rcm8cube.show_plan('velocity', t=40, ax=ax[1], ticks=True)
rcm8cube.show_plan('strata_sand_frac', t=40, ax=ax[2])
plt.show()
