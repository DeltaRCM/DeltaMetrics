golfcube = dm.sample_data.golf()

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
golfcube.show_plan('eta', t=40, ax=ax[0])
golfcube.show_plan('velocity', t=40, ax=ax[1], ticks=True)
golfcube.show_plan('sandfrac', t=40, ax=ax[2])
plt.show()
