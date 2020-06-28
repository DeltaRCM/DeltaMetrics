"""Visual for WetMask."""
import deltametrics as dm
from deltametrics.mask import WetMask

rcm8cube = dm.sample_data.cube.rcm8()
wet_mask = WetMask(rcm8cube['eta'][-1, :, :])
wet_mask.show()
