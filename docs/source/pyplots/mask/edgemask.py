"""Visual for EdgeMask."""
import deltametrics as dm
from deltametrics.mask import EdgeMask

golfcube = dm.sample_data.golf()
edge_mask = EdgeMask(golfcube['eta'].data[-1, :, :])
edge_mask.show()
