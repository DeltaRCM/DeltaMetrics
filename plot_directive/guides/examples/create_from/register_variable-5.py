test_data.register_variable("sqrt_new_var_spacetime", sqrt_new_var_spacetime)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(sqrt_new_var_spacetime[:, 20, :], origin="lower")
strike_data.show("sqrt_new_var_spacetime", ax=ax[1])
plt.show()