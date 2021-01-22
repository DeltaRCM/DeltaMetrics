locs = [0.25, 1, 0.5, 4, 2]
scales = [0.1, 0.25, 0.4, 0.5, 0.1]
bins = np.linspace(0, 6, num=40)

hist_bin_sets = [np.histogram(np.random.normal(l, s, size=500), bins=bins, density=True) for l, s in zip(locs, scales)]

fig, ax = plt.subplots()
dm.plot.show_histograms(*hist_bin_sets, sets=[0, 1, 0, 1, 2], ax=ax)
ax.set_xlim((0, 6))
ax.set_ylabel('density')
plt.show()