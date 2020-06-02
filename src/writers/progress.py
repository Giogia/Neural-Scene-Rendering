
import os
import numpy as np
from PIL import Image


def concatenate(image, rec):
    rows = []
    row = []

    for i in range(image.size(0)):
        row.append(
            np.concatenate((
                rec[i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                image[i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1))
        if len(row) == 2:
            rows.append(np.concatenate(row, axis=1))
            row = []

    out_image = np.concatenate(rows, axis=0)
    return Image.fromarray(np.clip(out_image, 0, 255).astype(np.uint8))


class ProgressWriter:
    def __init__(self, outpath):
        self.outpath = outpath

    def batch(self, iter_num, **kwargs):

        image = concatenate(kwargs["image"], kwargs["i_rgb_rec"])
        image.save(os.path.join(self.outpath, "prog_{:06}.jpg".format(iter_num)))

        depth = concatenate(kwargs["depth"], kwargs["i_depth_rec"])
        depth.save(os.path.join(self.outpath, "prog_{:06}_depth.jpg".format(iter_num)))


