fig, ax = plt.subplots(1, 3, figsize=(10, 4))
golfcube.show_plan('eta', t=-1, ax=ax[0])
im1 = ax[1].imshow(OAP.below_mask,
                   cmap='Greys_r', origin='lower')
im2 = ax[2].imshow(OAP.sea_angles,
                   cmap='jet', origin='lower')
dm.plot.append_colorbar(im2, ax=ax[2])
ax[0].set_title('input elevation data')
ax[1].set_title('OAP.below_mask')
ax[2].set_title('OAP.sea_angles')
for i in range(1, 3):
    ax[i].set_xticks([])
    ax[i].set_yticks([])