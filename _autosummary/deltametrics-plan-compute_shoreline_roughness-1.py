rcm8 = dm.sample_data.cube.rcm8()

# early in model run
lm0 = dm.mask.LandMask(rcm8['eta'][5, :, :])
sm0 = dm.mask.ShorelineMask(rcm8['eta'][5, :, :])

# late in model run
lm1 = dm.mask.LandMask(rcm8['eta'][-1, :, :])
sm1 = dm.mask.ShorelineMask(rcm8['eta'][-1, :, :])

# compute roughnesses
rgh0 = dm.plan.compute_shoreline_roughness(sm0, lm0)
rgh1 = dm.plan.compute_shoreline_roughness(sm1, lm1)

# make the plot
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
rcm8.show_plan('eta', t=5, ax=ax[0])
ax[0].set_title('roughness = {:.2f}'.format(rgh0))
rcm8.show_plan('eta', t=-1, ax=ax[1])
ax[1].set_title('roughness = {:.2f}'.format(rgh1))
plt.show()