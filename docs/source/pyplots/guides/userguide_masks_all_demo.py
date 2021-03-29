import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np

import deltametrics as dm

# use a new cube
maskcube = dm.sample_data.golf()

# create the masks from variables in the cube
land_mask = dm.mask.LandMask(
    maskcube['eta'][-1, :, :],
    elevation_threshold=0)

wet_mask = dm.mask.WetMask(
    maskcube['eta'][-1, :, :],
    elevation_threshold=0)

channel_mask = dm.mask.ChannelMask(
    maskcube['eta'][-1, :, :],
    maskcube['velocity'][-1, :, :],
    elevation_threshold=0,
    flow_threshold=0.3)

centerline_mask = dm.mask.CenterlineMask(
    maskcube['eta'][-1, :, :],
    maskcube['velocity'][-1, :, :],
    elevation_threshold=0,
    flow_threshold=0.3)

edge_mask = dm.mask.EdgeMask(
    maskcube['eta'][-1, :, :],
    elevation_threshold=0)

shore_mask = dm.mask.ShorelineMask(
    maskcube['eta'][-1, :, :],
    elevation_threshold=0)

fig = plt.figure(constrained_layout=True, figsize=(12, 10))
spec = gs.GridSpec(ncols=2, nrows=4, figure=fig)
ax0 = fig.add_subplot(spec[0, :])
axs = [fig.add_subplot(spec[i, j]) for i, j in zip(np.repeat(
    np.arange(1, 4), 2), np.tile(np.arange(2), (4,)))]
maskcube.show_plan('eta', t=-1, ax=ax0)

for i, m in enumerate([land_mask, wet_mask, channel_mask,
                       centerline_mask, edge_mask, shore_mask]):
    m.show(ax=axs[i])
    axs[i].set_title(m.mask_type)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
