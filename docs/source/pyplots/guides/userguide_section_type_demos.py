import matplotlib.gridspec as gs

rcm8cube = dm.sample_data.cube.rcm8()
rcm8cube.stratigraphy_from('eta')
rcm8cube.register_section('demo', dm.section.StrikeSection(y=10))

_strike = dm.section.StrikeSection(rcm8cube, y=18)
__path = np.column_stack((np.linspace(50, 150, num=4000, dtype=np.int),
                          np.linspace(10, 90, num=4000, dtype=np.int)))
_path = dm.section.PathSection(rcm8cube, path=__path)

fig = plt.figure(constrained_layout=True, figsize=(10, 8))
spec = gs.GridSpec(ncols=2, nrows=3, figure=fig)
ax0 = fig.add_subplot(spec[0, :])
axs = [fig.add_subplot(spec[i, j]) for i, j in zip(
    np.repeat(np.arange(1, 3), 2), np.tile(np.arange(2), (3,)))]

rcm8cube.show_plan('eta', t=-1, ax=ax0, ticks=True)
for i, s in enumerate([_strike, _path]):
    ax0.plot(s.trace[:, 0], s.trace[:, 1], 'r--')
    s.show('velocity', ax=axs[i])
    axs[i].set_title(s.section_type)
