rcm8 = dm.sample_data.cube.rcm8()

# early in model run
sm0 = dm.mask.ShorelineMask(rcm8['eta'][5, :, :])

# late in model run
sm1 = dm.mask.ShorelineMask(rcm8['eta'][-1, :, :])

# compute lengths
len0 = dm.plan.compute_shoreline_length(sm0)
len1, line1 = dm.plan.compute_shoreline_length(sm1, return_line=True)

# make the plot
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
rcm8.show_plan('eta', t=5, ax=ax[0])
ax[0].set_title('length = {:.2f}'.format(len0))
rcm8.show_plan('eta', t=-1, ax=ax[1])
ax[1].plot(line1[:, 0], line1[:, 1], 'r-')
ax[1].set_title('length = {:.2f}'.format(len1))
plt.show()