golfcube = dm.sample_data.golf()
arr = golfcube['eta'][-1, :, :].data
gmsk = dm.mask.GeometricMask(arr)

# Define an angular mask to cover part of the domain from 0 to pi/3.
gmsk.angular(0, np.pi/3)

# Visualize the mask:
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
gmsk.show(ax=ax[0], title='Binary Mask')
ax[1].imshow(golfcube['eta'][-1, :, :]*gmsk.mask, origin='lower')
ax[1].set_xticks([]); ax[1].set_yticks([])
ax[1].set_title('Mask * Topography')
plt.show()