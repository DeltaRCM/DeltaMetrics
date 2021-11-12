import matplotlib.pyplot as plt

import deltametrics as dm

golfcube = dm.sample_data.golf()
golfcube.stratigraphy_from('eta')
golfcube.register_section('demo', dm.section.StrikeSection(y=10))

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 3.5))

ax[0].imshow(golfcube.sections['demo']['velocity'],
             origin='lower', cmap=golfcube.varset['velocity'].cmap)
ax[0].set_ylabel('$t$ coordinate')

ax[1].imshow(golfcube.sections['demo']['velocity'].as_preserved(),
             origin='lower', cmap=golfcube.varset['velocity'].cmap)
ax[1].set_ylabel('$t$ coordinate')

ax[1].set_xlabel('$s$ coordinate')
plt.show()
