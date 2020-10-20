rcm8cube = dm.sample_data.cube.rcm8()

nt = 5
ts = np.linspace(0, rcm8cube['eta'].shape[0]-1, num=nt, dtype=np.int)

fig, ax = plt.subplots(1, nt, figsize=(12, 2))
for i, t in enumerate(ts):
    ax[i].imshow(rcm8cube['eta'].data[t, :, :], vmin=-5, vmax=0.5)
    ax[i].set_title('t = ' + str(t))
    ax[i].axes.get_xaxis().set_ticks([])
    ax[i].axes.get_yaxis().set_ticks([])
ax[0].set_ylabel('y-direction')
ax[0].set_xlabel('x-direction')
