lm0.trim_mask(length=golf.meta['L0'].data+1)
sm0.trim_mask(length=golf.meta['L0'].data+1)
lm1.trim_mask(length=golf.meta['L0'].data+1)
sm1.trim_mask(length=golf.meta['L0'].data+1)

fig, ax = plt.subplots(1, 2, figsize=(8, 3))
lm0.show(ax=ax[0])
sm0.show(ax=ax[1])
plt.show()