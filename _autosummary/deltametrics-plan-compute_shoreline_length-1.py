golf = dm.sample_data.golf()

# early in model run
sm0 = dm.mask.ShorelineMask(
    golf['eta'][15, :, :],
    elevation_threshold=0,
    elevation_offset=-0.5)

# late in model run
sm1 = dm.mask.ShorelineMask(
    golf['eta'][-1, :, :],
    elevation_threshold=0,
    elevation_offset=-0.5)

# compute lengths
len0 = dm.plan.compute_shoreline_length(sm0)
len1, line1 = dm.plan.compute_shoreline_length(sm1, return_line=True)

# make the plot
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
golf.show_plan('eta', t=15, ax=ax[0])
ax[0].set_title('length = {:.2f}'.format(len0))
golf.show_plan('eta', t=-1, ax=ax[1])
ax[1].plot(line1[:, 0], line1[:, 1], 'r-')
ax[1].set_title('length = {:.2f}'.format(len1))
plt.show()