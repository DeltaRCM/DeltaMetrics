rcm8cube = dm.sample_data.cube.rcm8()

nt = 5
ts = np.linspace(0, rcm8cube['eta'].shape[0]-1, num=nt, dtype=np.int)

# compute the change in bed elevation between the last two intervals above
diff_time = rcm8cube['eta'][ts[-1], ...] - rcm8cube['eta'][ts[-2], ...]
fig, ax = plt.subplots(figsize=(5, 3))
im = ax.imshow(diff_time, cmap='RdBu',
               vmax=abs(diff_time).max(),
               vmin=-abs(diff_time).max())
dm.plot.append_colorbar(im, ax)
