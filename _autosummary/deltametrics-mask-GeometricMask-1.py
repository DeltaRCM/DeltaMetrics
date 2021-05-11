golfcube = dm.sample_data.golf()
arr = golfcube['eta'][-1, :, :]
gmsk = dm.mask.GeometricMask(arr)

# Define an angular mask to cover half the domain from 0 to pi/2.
gmsk.angular(0, np.pi/2)

# Further mask this region by defining bounds in the strike direction.
gmsk.strike(10, 50)

# Visualize the mask:
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
gmsk.show(ax=ax[0])
ax[1].imshow(golfcube['eta'][-1, :, :]*gmsk.mask, origin='lower')
ax[1].set_xticks([]); ax[1].set_yticks([])
plt.show()