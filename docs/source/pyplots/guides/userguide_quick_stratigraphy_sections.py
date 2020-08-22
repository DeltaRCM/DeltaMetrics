rcm8cube = dm.sample_data.cube.rcm8()
rcm8cube.stratigraphy_from('eta')
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))


fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
rcm8cube.show_section('demo', 'velocity', ax=ax[0])
rcm8cube.show_section('demo', 'velocity',
                      display_array_style='preserved', ax=ax[1])
rcm8cube.show_section('demo', 'velocity',
                      display_array_style='stratigraphy', ax=ax[2])
