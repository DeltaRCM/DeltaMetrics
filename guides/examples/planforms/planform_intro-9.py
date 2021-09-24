SM_from_OAM = dm.mask.ShorelineMask.from_Planform(
  OAP, contour_threshold=75)

SM_from_MPM = dm.mask.ShorelineMask.from_Planform(
  MP, contour_threshold=0.75)

fig, ax = plt.subplots(1, 3, figsize=(10, 5), dpi=300)

ax[0].imshow(SM_from_OAM.mask, interpolation=None)
ax[0].set_title('Shoreline from OAM')

ax[1].imshow(SM_from_MPM.mask, interpolation=None)
ax[1].set_title('Shoreline from MPM')

d_plot = ax[2].imshow(
  SM_from_OAM.mask.astype(float) - SM_from_MPM.mask.astype(float),
  interpolation=None, cmap='bone')
ax[2].set_title('OAM shoreline - MPM shoreline')
plt.colorbar(d_plot, ax=ax[2], fraction=0.05)

plt.tight_layout()
plt.show()