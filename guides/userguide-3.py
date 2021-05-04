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