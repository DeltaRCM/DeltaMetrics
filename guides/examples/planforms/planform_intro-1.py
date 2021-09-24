golfcube = dm.sample_data.golf()

plt.imshow(golfcube['eta'][-1, :, :])
plt.colorbar()
plt.title('Final Elevation Data')
plt.show()