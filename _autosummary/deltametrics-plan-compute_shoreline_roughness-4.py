# compute roughnesses
rgh0 = dm.plan.compute_shoreline_roughness(sm0, lm0)
rgh1 = dm.plan.compute_shoreline_roughness(sm1, lm1)

# make the plot
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
golf.show_plan('eta', t=15, ax=ax[0])
ax[0].set_title('roughness = {:.2f}'.format(rgh0))
golf.show_plan('eta', t=-1, ax=ax[1])
ax[1].set_title('roughness = {:.2f}'.format(rgh1))
plt.show()