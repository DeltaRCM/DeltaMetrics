golfcube = dm.sample_data.golf()
golfcube.stratigraphy_from('eta')
golfcube.register_section('demo', dm.section.StrikeSection(distance_idx=10))

fig, ax = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
_v = 'velocity'
golfcube.sections['demo'].show(
    _v, style='lines', data='spacetime', ax=ax[0, 0])
golfcube.sections['demo'].show(
    _v, style='shaded', data='spacetime', ax=ax[0, 1])
golfcube.sections['demo'].show(
    _v, style='lines', data='preserved', ax=ax[1, 0])
golfcube.sections['demo'].show(
    _v, style='shaded', data='preserved', ax=ax[1, 1])
golfcube.sections['demo'].show(
    _v, style='lines', data='stratigraphy', ax=ax[2, 0])
golfcube.sections['demo'].show(
    _v, style='shaded', data='stratigraphy', ax=ax[2, 1])
plt.show(block=False)
