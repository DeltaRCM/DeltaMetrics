"""Visual for ChannelMask."""
import deltametrics as dm
from deltametrics.mask import ChannelMask

golfcube = dm.sample_data.golf()
channel_mask = ChannelMask(golfcube['velocity'].data[-1, :, :],
                           golfcube['eta'].data[-1, :, :])
channel_mask.show()
