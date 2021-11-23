import matplotlib.gridspec as gs

golfcube = dm.sample_data.golf()

_strike = dm.section.StrikeSection(
    golfcube, distance=1200)
_dip = dm.section.DipSection(
    golfcube, distance=3000)
_path = dm.section.PathSection(
    golfcube, path=np.array([[1400, 2000], [2000, 4000], [3000, 6000]]))
_circ = dm.section.CircularSection(
    golfcube, radius=2000)
_rad = dm.section.RadialSection(
    golfcube, azimuth=70)

fig = plt.figure(constrained_layout=False, figsize=(10, 9))
spec = gs.GridSpec(ncols=2, nrows=4, figure=fig, wspace=0.3, hspace=0.4,
                   left=0.1, right=0.9, top=0.95, bottom=0.05,
                   height_ratios=[1.5, 1, 1, 1])
ax0 = fig.add_subplot(spec[0, :])
axs = [fig.add_subplot(spec[i, j]) for i, j in zip(
    np.repeat(np.arange(1, 4), 2), np.tile(np.arange(2), (4,)))]

t10cmap = plt.cm.get_cmap('tab10')
golfcube.show_plan('eta', t=-1, ax=ax0, ticks=True)
for i, s in enumerate([_strike, _dip, _path, _circ, _rad]):
    s.show_trace('--', color=t10cmap(i), ax=ax0)
    s.show('velocity', ax=axs[i])
    axs[i].set_title(s.section_type, color=t10cmap(i))

fig.delaxes(axs[-1])
