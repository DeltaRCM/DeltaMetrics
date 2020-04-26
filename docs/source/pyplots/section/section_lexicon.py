import matplotlib.pyplot as plt

import deltametrics as dm

rcm8cube = dm.sample_data.cube.rcm8()
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))

fig, ax = plt.subplots(figsize=(6, 2))
ax.imshow(rcm8cube.sections['demo']['strata_sand_frac'].as_spacetime(),
          origin='lower', cmap=rcm8cube.varset.strata_sand_frac.cmap)
ax.set_xlabel('$s$ coordinate')
ax.set_ylabel('$z$ coordinate')
plt.show()
