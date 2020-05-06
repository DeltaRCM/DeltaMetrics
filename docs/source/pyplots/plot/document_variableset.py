import matplotlib.pyplot as plt
import numpy as np

import matplotlib as mpl

import deltametrics as dm


vs = dm.plot.VariableSet()

N = 256
gradient = np.linspace(0, 1, N)
gradient = np.vstack((gradient, gradient))


def over_coordinates(ax_lims):
    tri_low = np.array([[0, 0],
                        [0 - 10, 0.5],
                        [0, 1]])
    tri_high = np.array([[N, 0],
                         [N + 10, 0.5],
                         [N, 1]])
    name_xy = np.array([-240, 0.8])
    label_xy = np.array([-240, 0.4])

    return tri_low, tri_high, name_xy, label_xy


def show_an_info(Info, ax):
    """Show a specific info.
    """
    ax_lims = ax.get_position().get_points()
    tri_low, tri_high, name_xy, label_xy = over_coordinates(ax_lims)

    ax.imshow(gradient, aspect='auto', cmap=Info.cmap)
    if not Info.norm:
        ax.set_xticks([])
    else:
        ax.set_xticks([Info.vmin])
        ax.add_patch(mpl.patches.Polygon(tri_low, closed=True,
                     color=Info.cmap._rgba_over, clip_on=False))
        ax.add_patch(mpl.patches.Polygon(tri_high, closed=True,
                     color=Info.cmap._rgba_over, clip_on=False))

    ax.text(name_xy[0], name_xy[1], 'name: ' + Info.name, fontsize=8,
            horizontalalignment='left', verticalalignment='bottom')
    ax.text(label_xy[0], label_xy[1], 'label: ' + Info.label, fontsize=8,
            horizontalalignment='left', verticalalignment='bottom')

    ax.set_yticks([])
    ax.set_ylim([0, 1])
    ax.tick_params(labelsize=8)

nvi = len(vs.known_list)
fig, ax = plt.subplots(nvi, 1, figsize=(8, 8))
plt.tight_layout()
plt.subplots_adjust(left=0.487, right=0.95)
ax = ax.flatten()
for i, v in enumerate(vs.known_list):
    show_an_info(vs[v], ax[i])

plt.show()
