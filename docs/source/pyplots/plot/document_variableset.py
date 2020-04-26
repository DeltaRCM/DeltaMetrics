import matplotlib.pyplot as plt


def show_an_info(Info):
    """Show a specific info.
    """
    fig, ax = plt.subplots(figsize=(4, 0.5))
    ax.imshow(gradient, aspect='auto', cmap=Info.cmap)
    plt.show()


def show_net_to_gross():
    show_an_info(vs.net_to_gross)


def show_time():
    show_an_info(vs.time)


def showeta():
    """The one"""
    import numpy as np
    import deltametrics as dm

    vs = dm.plot.VariableSet()
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))

    plt.figure()
    plt.plot(range(6))
    plt.show()


def show_stage():
    show_an_info(vs.stage)


def show_depth():
    show_an_info(vs.depth)


def show_discharge():
    show_an_info(vs.discharge)


def show_velocity():
    show_an_info(vs.velocity)


def show_strata_age():
    show_an_info(vs.strata_age)


def show_strata_sand_frac():
    show_an_info(vs.strata_sand_frac)


def show_strata_depth():
    show_an_info(vs.strata_depth)
