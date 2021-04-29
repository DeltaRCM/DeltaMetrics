golfcube = dm.sample_data.golf()
elevation_data = golfcube['eta'][-1, :, :]

fig, ax = plt.subplots()
dm.plot.aerial_view( elevation_data, ax=ax)
plt.show()