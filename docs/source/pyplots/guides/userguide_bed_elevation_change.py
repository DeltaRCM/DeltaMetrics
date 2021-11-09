golfcube = dm.sample_data.golf()

nt = 5
ts = np.linspace(0, golfcube['eta'].shape[0]-1, num=nt, dtype=np.int)

# compute the change in bed elevation between the last two intervals above
diff_time = golfcube['eta'].data[ts[-1], ...] - golfcube['eta'].data[ts[-2], ...]
fig, ax = plt.subplots(figsize=(5, 3))
im = ax.imshow(diff_time, cmap='RdBu',
               vmax=abs(diff_time).max(),
               vmin=-abs(diff_time).max())
dm.plot.append_colorbar(im, ax)
