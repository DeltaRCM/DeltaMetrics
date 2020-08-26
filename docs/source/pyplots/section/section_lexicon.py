import matplotlib.pyplot as plt

import deltametrics as dm

rcm8cube = dm.sample_data.cube.rcm8()
rcm8cube.stratigraphy_from('eta')
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 3.5))

ax[0].imshow(rcm8cube.sections['demo']['velocity'],
             origin='lower', cmap=rcm8cube.varset['velocity'].cmap)
ax[0].set_ylabel('$t$ coordinate')

ax[1].imshow(rcm8cube.sections['demo']['velocity'].as_preserved(),
             origin='lower', cmap=rcm8cube.varset['velocity'].cmap)
ax[1].set_ylabel('$t$ coordinate')

ax[1].set_xlabel('$s$ coordinate')
plt.show()
