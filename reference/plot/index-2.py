golfcube = dm.sample_data.golf()
elevation_data = golfcube['eta'][-1, :, :]
sparse_data = (golfcube['discharge'][-1, ...] /
              (golfcube.meta['h0'].data *
               golfcube.meta['u0'][-1].data))

fig, ax = plt.subplots(1, 3, figsize=(8, 3))
for axi in ax.ravel():
    dm.plot.aerial_view( elevation_data, ax=axi)

dm.plot.overlay_sparse_array(
    sparse_data, ax=ax[0])  # default clip is (None, 90)
dm.plot.overlay_sparse_array(
    sparse_data, alpha_clip=(None, None), ax=ax[1])
dm.plot.overlay_sparse_array(
    sparse_data, alpha_clip=(70, 90), ax=ax[2])

plt.tight_layout()
plt.show()