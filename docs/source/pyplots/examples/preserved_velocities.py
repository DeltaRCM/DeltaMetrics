import numpy as np
import deltametrics as dm
import matplotlib.pyplot as plt

golfcube = dm.sample_data.golf()
golfcube.stratigraphy_from('eta')


def pick_velocities(sect):
    """Extract data.

    Utility function to grab mean velocity of a section, returning the value
    from the total section, and just the preserved section.

    Note that both extractions are limited to where sediment has been
    deposited/eroded in the model domain.
    """
    _whr = [sect['eta'] != sect['eta'][0, :]]
    _a = np.nanmean(sect['velocity'].data[tuple(_whr)])
    _s = np.nanmean(sect['velocity'].strat.as_preserved().data[tuple(_whr)])
    return _a, _s


# preallocate
_ys = np.arange(10, 90, step=2)  # sections to examine
_mall = np.full_like(_ys, np.nan, dtype=float)
_mstrat = np.full_like(_ys, np.nan, dtype=float)


# loop through all of the sections defined in _ys
for i, _y in enumerate(_ys):
    _s = dm.section.StrikeSection(golfcube, distance_idx=_y)
    _mall[i], _mstrat[i] = pick_velocities(_s)


# make the plot
fig, ax = plt.subplots()
ax.plot(_mall, 'b-', label='all values')
ax.plot(_mstrat, 'r-', label='preserved only')
ax.set_xlabel('distance from channel inlet')
ax.set_ylabel('mean velocity')
ax.legend()
plt.show()
