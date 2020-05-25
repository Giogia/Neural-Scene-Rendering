import numpy as np
from Imath import PixelType
from OpenEXR import InputFile


def exr_to_image(path):
    file = InputFile(path)
    window = file.header()['dataWindow']
    channels = ('R', 'G', 'B')
    size = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)

    channels_tuple = [np.frombuffer(channel, dtype=np.float32)
                      for channel in file.channels(channels, PixelType(PixelType.FLOAT))]
    exr_array = np.dstack(channels_tuple)
    return exr_array.reshape(size + (len(channels_tuple),))


def exr_to_depth(path, far_threshold=np.inf):
    file = InputFile(path)
    window = file.header()['dataWindow']
    size = (window.max.y - window.min.y + 1, window.max.x - window.min.x + 1)

    exr_depth = file.channel('Z', PixelType(PixelType.FLOAT))
    exr_depth = np.fromstring(exr_depth, dtype=np.float32)
    exr_depth[exr_depth > far_threshold] = 0
    exr_depth = np.reshape(exr_depth, size)

    return exr_depth