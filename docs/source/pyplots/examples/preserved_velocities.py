import numpy as np
import deltametrics as dm
import matplotlib.pyplot as plt

rcm8cube = dm.sample_data.rcm8()
rcm8cube.stratigraphy_from('eta')


def pick_velocities(sect):
    """Extract data.

    Utility function to grab mean velocity of a section, returning the value
    from the total section, and just the preserved section.

    Note that both extractions are limited to where sediment has been
    deposited/eroded in the model domain.
    """
    _whr = [sect['eta'] != sect['eta'][0, :]]
    _a = np.nanmean(sect['velocity'][tuple(_whr)])
    _s = np.nanmean(sect['velocity'].as_preserved()[tuple(_whr)])
    return _a, _s


# preallocate
_ys = np.arange(10, 110, step=2)  # sections to examine
_mall = np.full_like(_ys, np.nan, dtype=np.float)
_mstrat = np.full_like(_ys, np.nan, dtype=np.float)


# loop through all of the sections defined in _ys
for i, _y in enumerate(_ys):
    _s = dm.section.StrikeSection(rcm8cube, y=_y)
    _mall[i], _mstrat[i] = pick_velocities(_s)


# make the plot
fig, ax = plt.subplots()
ax.plot(_mall, 'b-', label='all values')
ax.plot(_mstrat, 'r-', label='preserved only')
ax.set_xlabel('distance from channel inlet')
ax.set_ylabel('mean velocity')
ax.legend()
plt.show()
