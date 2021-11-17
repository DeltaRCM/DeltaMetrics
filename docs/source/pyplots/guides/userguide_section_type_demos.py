import matplotlib.gridspec as gs

golfcube = dm.sample_data.golf()
golfcube.stratigraphy_from('eta')
golfcube.register_section('demo', dm.section.StrikeSection(idx=10))

_strike = dm.section.StrikeSection(golfcube, y=18)
__path = np.column_stack((np.linspace(10, 90, num=4000, dtype=int),
                          np.linspace(50, 150, num=4000, dtype=int)))
_path = dm.section.PathSection(golfcube, path=__path)
_circular = dm.section.CircularSection(golfcube, radius=40)
_rad = dm.section.RadialSection(golfcube, azimuth=70)

fig = plt.figure(constrained_layout=True, figsize=(10, 8))
spec = gs.GridSpec(ncols=2, nrows=3, figure=fig, wspace=0.2, hspace=0.2)
ax0 = fig.add_subplot(spec[0, :])
axs = [fig.add_subplot(spec[i, j]) for i, j in zip(
    np.repeat(np.arange(1, 3), 2), np.tile(np.arange(2), (3,)))]

golfcube.show_plan('eta', t=-1, ax=ax0, ticks=True)
for i, s in enumerate([_strike, _path, _circular, _rad]):
    s.show_trace('--', ax=ax0)
    s.show('velocity', ax=axs[i])
    axs[i].set_title(s.section_type)
