import matplotlib.gridspec as gs

rcm8cube = dm.sample_data.cube.rcm8()

land_mask = dm.mask.LandMask(rcm8cube['eta'][-1, :, :])
wet_mask = dm.mask.WetMask(rcm8cube['eta'][-1, :, :])
channel_mask = dm.mask.ChannelMask(rcm8cube['velocity'][-1, :, :], rcm8cube['eta'][-1, :, :])
centerline_mask = dm.mask.CenterlineMask(channel_mask)
edge_mask = dm.mask.EdgeMask(rcm8cube['eta'][-1, :, :])
shore_mask = dm.mask.ShorelineMask(rcm8cube['eta'][-1, :, :])

fig = plt.figure(constrained_layout=True, figsize=(12, 10))
spec = gs.GridSpec(ncols=2, nrows=4, figure=fig)
ax0 = fig.add_subplot(spec[0, :])
axs = [fig.add_subplot(spec[i, j]) for i, j in zip(np.repeat(
    np.arange(1, 4), 2), np.tile(np.arange(2), (4,)))]
ax0.imshow(rcm8cube['eta'][-1, :, :])

for i, m in enumerate([land_mask, wet_mask, channel_mask,
                       centerline_mask, edge_mask, shore_mask]):
    axs[i].imshow(m.mask[-1, :, :], cmap='gray')
    axs[i].set_title(m.mask_type)
