rcm8cube = dm.sample_data.cube.rcm8()
rcm8cube.stratigraphy_from('eta')
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))

fig, ax = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
_v = 'velocity'
rcm8cube.sections['demo'].show(
    _v, style='lines', display_array_style='spacetime', ax=ax[0, 0])
rcm8cube.sections['demo'].show(
    _v, style='shaded', display_array_style='spacetime', ax=ax[0, 1])
rcm8cube.sections['demo'].show(
    _v, style='lines', display_array_style='preserved', ax=ax[1, 0])
rcm8cube.sections['demo'].show(
    _v, style='shaded', display_array_style='preserved', ax=ax[1, 1])
rcm8cube.sections['demo'].show(
    _v, style='lines', display_array_style='stratigraphy', ax=ax[2, 0])
rcm8cube.sections['demo'].show(
    _v, style='shaded', display_array_style='stratigraphy', ax=ax[2, 1])
plt.show(block=False)
