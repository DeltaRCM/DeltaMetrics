golfcube = dm.sample_data.golf()

OAP = dm.plan.OpeningAnglePlanform.from_elevation_data(
    golfcube['eta'][-1, :, :],
    elevation_threshold=0)

lm = dm.mask.LandMask.from_OAP(OAP)
sm = dm.mask.ShorelineMask.from_OAP(OAP)

fig, ax = plt.subplots(2, 2)
golfcube.show_plan('eta', t=-1, ax=ax[0, 0])
ax[0, 1].imshow(OAP.sea_angles, vmax=180, cmap='jet')
lm.show(ax=ax[1, 0])
sm.show(ax=ax[1, 1])
