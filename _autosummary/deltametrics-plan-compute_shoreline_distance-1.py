golf = dm.sample_data.golf()

sm = dm.mask.ShorelineMask(
    golf['eta'][-1, :, :],
    elevation_threshold=0,
    elevation_offset=-0.5)

# compute mean and stddev distance
mean, stddev = dm.plan.compute_shoreline_distance(
    sm, origin=[golf.meta['CTR'].data, golf.meta['L0'].data])

# make the plot
fig, ax = plt.subplots()
golf.show_plan('eta', t=-1, ticks=True, ax=ax)
ax.set_title('mean = {:.2f}'.format(mean))
plt.show()