rcm8cube = dm.sample_data.rcm8()
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 6))
rcm8cube.show_section('demo', 'eta', ax=ax[0])
rcm8cube.show_section('demo', 'velocity', ax=ax[1])
rcm8cube.show_section('demo', 'strata_sand_frac', ax=ax[2])
