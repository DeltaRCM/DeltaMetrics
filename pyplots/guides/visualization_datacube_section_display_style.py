rcm8cube = dm.sample_data.rcm8()
rcm8cube.stratigraphy_from('eta')
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))

fig, ax = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
_v = 'velocity'
rcm8cube.sections['demo'].show(
    _v, style='lines', data='spacetime', ax=ax[0, 0])
rcm8cube.sections['demo'].show(
    _v, style='shaded', data='spacetime', ax=ax[0, 1])
rcm8cube.sections['demo'].show(
    _v, style='lines', data='preserved', ax=ax[1, 0])
rcm8cube.sections['demo'].show(
    _v, style='shaded', data='preserved', ax=ax[1, 1])
rcm8cube.sections['demo'].show(
    _v, style='lines', data='stratigraphy', ax=ax[2, 0])
rcm8cube.sections['demo'].show(
    _v, style='shaded', data='stratigraphy', ax=ax[2, 1])
plt.show(block=False)
