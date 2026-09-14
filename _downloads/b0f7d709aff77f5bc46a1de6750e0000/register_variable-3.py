# Register the square root of the variable directly to the StratigraphyCube
test_strat.register_variable("sqrt_new_var", np.sqrt(test_strat["new_var"]))

# make a strike section
strike_strat = spl.section.StrikeSection(test_strat, distance_idx=20)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(test_strat["sqrt_new_var"][:, 20, :], origin="lower")
strike_strat.show("sqrt_new_var", ax=ax[1])
plt.show()