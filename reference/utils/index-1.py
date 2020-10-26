from deltametrics.utils import line_to_cells
p0 = (1, 6)
p1 = (6, 3)
x, y = line_to_cells(p0, p1)

fig, ax = plt.subplots(figsize=(2, 2))
_arr = np.zeros((10, 10))
_arr[y, x] = 1
ax.imshow(_arr, cmap='gray')
ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-')
plt.show()
