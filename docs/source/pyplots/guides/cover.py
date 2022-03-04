import deltametrics as dm
import matplotlib.pyplot as plt


golfcube = dm.sample_data.golf()
golfstrat = dm.cube.StratigraphyCube.from_DataCube(
    golfcube, dz=0.05)

plan = dm.plan.Planform(golfcube, idx=-1)
sect = dm.section.StrikeSection(golfstrat, distance_idx=20)

_labelsize = 7
_ticksize = 7


fig, ax = plt.subplots(
  2, 1, figsize=(5, 4),
  gridspec_kw={'wspace': 0.4,
               'left': 0.07, 'right': 0.95})

plan.show('eta', ax=ax[0], ticks=True)
sect.show_trace('r--', ax=ax[0])

cbar = ax[0].images[-1].colorbar
cbar.set_label('elevation (m)', fontsize=_labelsize)
cbar.ax.tick_params(labelsize=_ticksize)

ax[0].set_xlabel('distance (m)', fontsize=_labelsize)
ax[0].set_ylabel('distance (m)', fontsize=_labelsize)


sect.show('time', ax=ax[1])

cbar = ax[1].collections[-1].colorbar
cbar.set_label('time\ndeposited (s)', fontsize=_labelsize)
cbar.ax.tick_params(labelsize=_ticksize)
cbar.ax.yaxis.offsetText.set_fontsize(_ticksize)


ax[1].set_xlabel('along section distance (m)', fontsize=_labelsize)
ax[1].set_ylabel('elevation (m)', fontsize=_labelsize)

ax[1].set_position([0.1, 0.1, 0.8, 0.2])

for axi in ax.ravel():
    axi.tick_params(labelsize=_ticksize)

plt.show()
