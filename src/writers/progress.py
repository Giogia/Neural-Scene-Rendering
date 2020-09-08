import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def concatenate(rec, image=None):

    rows = []
    row = []

    if image is not None:

        for i in range(image.size(0)):
            row.append(
                np.concatenate((
                    rec[i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                    image[i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]),
                    axis=1))

            if len(row) == 2:
                rows.append(np.concatenate(row, axis=1))
                row = []

        return np.concatenate(rows, axis=0)

    else:

        for i in range(rec.size(0)):
            row.append(rec[i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2])

            if len(row) == 4:
                rows.append(np.concatenate(row, axis=1))
                row = []

        return np.concatenate(rows, axis=0)


def recolor(image, colors='magma'):

    color_map = plt.get_cmap(colors)
    image = color_map(image[:, :, 0] / np.max(image))

    return 255 * image[:, :, :3]


class ProgressWriter:
    def __init__(self, outpath):
        self.outpath = outpath
        self.tensorboard = SummaryWriter(self.outpath)

    def batch(self, iter_num, ground_truth=True, **kwargs):

        if ground_truth:
            image = concatenate(kwargs['i_rgb_rec'], kwargs['image'])
        else:
            image = concatenate(kwargs['i_rgb_rec'])
        image = np.clip(image, 0, 255).astype(np.uint8)
        self.tensorboard.add_image('predicted colors', image, global_step=iter_num, dataformats='HWC')
        Image.fromarray(image).save(os.path.join(self.outpath, "prog_{:06}.jpg".format(iter_num)))

        if 'depth' in kwargs.keys():

            depth = concatenate(kwargs['i_depth_rec'], kwargs['depth'])
            depth = recolor(depth)
            depth = np.clip(depth, 0, 255).astype(np.uint8)
            self.tensorboard.add_image('predicted depth', depth, global_step=iter_num, dataformats='HWC')
            Image.fromarray(depth).save(os.path.join(self.outpath, "prog_{:06}_depth.jpg".format(iter_num)))

