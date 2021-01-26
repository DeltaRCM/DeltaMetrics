landsat = dm.sample_data.landsat()
nt = landsat.shape[0]

maxr = np.max(landsat['Red'][:])
maxg = np.max(landsat['Green'][:])
maxb = np.max(landsat['Blue'][:])

fig, ax = plt.subplots(1, nt, figsize=(12, 2))
for i in np.arange(nt):
    _arr = np.dstack((landsat['Red'][i, :, :]/maxr,
                      landsat['Green'][i, :, :]/maxg,
                      landsat['Blue'][i, :, :]/maxb))
    ax[i].imshow(_arr)
    ax[i].set_title('year = ' + str(landsat.t[i]))
    ax[i].axes.get_xaxis().set_ticks([])
    ax[i].axes.get_yaxis().set_ticks([])

plt.show()