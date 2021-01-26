"""Visual for CenterlineMask."""
import deltametrics as dm
from deltametrics.mask import ChannelMask
from deltametrics.mask import CenterlineMask

rcm8cube = dm.sample_data.rcm8()
channel_mask = ChannelMask(rcm8cube['velocity'].data[-1, :, :],
                           rcm8cube['eta'].data[-1, :, :])
centerline_mask = CenterlineMask(channel_mask)
centerline_mask.show()
