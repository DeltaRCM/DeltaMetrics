import matplotlib.pyplot as plt

import deltametrics as dm

golfcube = dm.sample_data.golf()
golfcube.stratigraphy_from('eta')
golfcube.register_section('demo', dm.section.StrikeSection(distance_idx=10))

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(6, 4))

golfcube.sections['demo'].show('velocity', ax=ax[0])
ax[0].set_ylabel('$t$ coordinate')

golfcube.sections['demo'].show('velocity', data='preserved', ax=ax[1])
ax[1].set_ylabel('$t$ coordinate')

ax[1].set_xlabel('$s$ coordinate')

plt.tight_layout()
plt.show()
