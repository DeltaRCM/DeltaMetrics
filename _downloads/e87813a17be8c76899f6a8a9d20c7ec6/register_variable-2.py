# make a strike section
strike_data = spl.section.StrikeSection(test_data, distance_idx=20)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(test_data["new_var"][:, 20, :], origin="lower")
strike_data.show("new_var", ax=ax[1])
plt.show()