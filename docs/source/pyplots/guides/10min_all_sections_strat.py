import deltametrics as dm
import matplotlib.pyplot as plt

golfcube = dm.sample_data.golf()
golfcube.stratigraphy_from('eta', dz=0.05)
golfcube.register_section('demo', dm.section.StrikeSection(distance_idx=10))

fig, ax = plt.subplots(5, 1, sharex=True, figsize=(6, 8))
ax = ax.flatten()
for i, var in enumerate(['time', 'eta', 'velocity', 'discharge', 'sandfrac']):
    golfcube.show_section('demo', var, data='stratigraphy',
                          ax=ax[i], label=True)
plt.show()
