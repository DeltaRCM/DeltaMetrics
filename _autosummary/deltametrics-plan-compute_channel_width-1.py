# set up the cube, mask, and section
golf = dm.sample_data.golf()
cm = dm.mask.ChannelMask(
    golf['eta'][-1, :, :],
    golf['velocity'][-1, :, :],
    elevation_threshold=0,
    flow_threshold=0.3)
sec = dm.section.CircularSection(golf, radius=40)

# compute the metric
m, s, w = dm.plan.compute_channel_width(
    cm, section=sec, return_widths=True)

fig, ax = plt.subplots()
cm.show(ax=ax)
sec.show_trace('r-', ax=ax)
ax.set_title(f'mean: {m:.2f}; stddev: {s:.2f}')
plt.show()