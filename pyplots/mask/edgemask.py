"""Visual for EdgeMask."""
import deltametrics as dm
from deltametrics.mask import EdgeMask

rcm8cube = dm.sample_data.cube.rcm8()
edge_mask = EdgeMask(rcm8cube['eta'].data[-1, :, :])
edge_mask.show()
