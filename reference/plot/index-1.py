fig, ax = plt.subplots(figsize=(5, 2))

# initial color red
red = (1.0, 0.0, 0.0)
ax.plot(-1, 1, 'o', color=red)

# scale from 1 to 0.05
scales = np.arange(1, 0, -0.05)

# loop through scales and plot
for s, scale in enumerate(scales):
    darker_red = dm.plot._scale_lightness(red, scale)
    ax.plot(s, scale, 'o', color=darker_red)

plt.show()