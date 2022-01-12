golfcube = dm.sample_data.golf()
golfcube.stratigraphy_from('eta')
golfcube.register_section('demo', dm.section.StrikeSection(distance_idx=10))


fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
golfcube.show_section('demo', 'velocity', ax=ax[0])
golfcube.show_section('demo', 'velocity',
                      data='preserved', ax=ax[1])
golfcube.show_section('demo', 'velocity',
                      data='stratigraphy', ax=ax[2])
