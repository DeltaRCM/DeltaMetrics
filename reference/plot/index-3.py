rcm8cube = dm.sample_data.cube.rcm8()

cmap0, norm0 = dm.plot.cartographic_colormap(H_SL=0)
cmap1, norm1 = dm.plot.cartographic_colormap(H_SL=0, h=5, n=0.5)

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
im0 = ax[0].imshow(rcm8cube['eta'][-1, ...], origin='lower',
               cmap=cmap0, norm=norm0)
cb0 = dm.plot.append_colorbar(im0, ax[0])
im1 = ax[1].imshow(rcm8cube['eta'][-1, ...], origin='lower',
               cmap=cmap1, norm=norm1)
cb1 = dm.plot.append_colorbar(im1, ax[1])
plt.show()