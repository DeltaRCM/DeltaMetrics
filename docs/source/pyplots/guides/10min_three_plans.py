golfcube = dm.sample_data.golf()

fig, ax = plt.subplots(1, 3, figsize=(10, 3))
golfcube.quick_show('eta', idx=40, ax=ax[0])
golfcube.quick_show('velocity', idx=40, ax=ax[1], ticks=True)
golfcube.quick_show('sandfrac', idx=40, ax=ax[2])
plt.show()
