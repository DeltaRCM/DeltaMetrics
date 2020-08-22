rcm8cube = dm.sample_data.cube.rcm8()
rcm8cube.stratigraphy_from('eta')

nt = 6
ts = np.linspace(0, rcm8cube['eta'].shape[0]-1, num=nt, dtype=np.int)

sc8cube = dm.cube.StratigraphyCube.from_DataCube(rcm8cube, dz=0.05)


def plot_section_type_demos():
    fig = plt.figure(constrained_layout=True, figsize=(10, 8))
    spec = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
    ax0 = fig.add_subplot(spec[0, :])
    axs = [fig.add_subplot(spec[i, j]) for i, j in zip(np.repeat(np.arange(1, 3), 2), np.tile(np.arange(2), (3,)))]

    rcm8cube.show_plan('eta', t=-1, ax=ax0, ticks=True)
    for i, s in enumerate([_strike, _path]):
        ax0.plot(s.trace[:, 0], s.trace[:, 1], 'r--')
        s.show('velocity', ax=axs[i])
        axs[i].set_title(s.section_type)


def plot_three_stratigraphy():
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(12, 8))
    rcm8cube.sections['demo'].show('time', style='lines',
                                   display_array_style='stratigraphy',
                                   ax=ax[0], label=True)
    sc8cube.sections['demo'].show('time', ax=ax[1], label='TIME')
    rcm8cube.sections['demo'].show('time', display_array_style='stratigraphy',
                                   ax=ax[2])


def plot_all_vars_stratigraphy():
    fig, ax = plt.subplots(7, 1, sharex=True, sharey=True, figsize=(12, 12))
    ax = ax.flatten()
    for i, var in enumerate(['time'] + rcm8cube.dataio.known_variables):
        sc8cube.show_section('demo', var, ax=ax[i], label=True,
                             style='shaded', display_array_style='stratigraphy')
    plt.show()  #doctest: +SKIP
