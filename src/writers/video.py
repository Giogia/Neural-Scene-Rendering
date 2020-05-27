
import multiprocessing
import os
import shutil
import subprocess

import matplotlib.cm as cm
import numpy as np
from PIL import Image


def writeimage(x):
    random_id, item_num, output_image = x

    output_image = np.clip(np.clip(output_image / 255., 0., 255.) ** (1. / 1.8) * 255., 0., 255).astype(np.uint8)

    if output_image.shape[1] % 2 != 0:
        output_image = output_image[:, :-1]

    Image.fromarray(output_image).save("/tmp/{}/{:06}.jpg".format(random_id, item_num))


class Writer():
    def __init__(self, outpath, show_target=False, show_diff=False, background_color=None, color_correction=None, n_threads=16):

        self.outpath = outpath
        self.show_target = show_target
        self.show_diff = show_diff
        self.background_color = np.array([0.5, 0.5, 0.5] if background_color is None else background_color, dtype=np.float32)
        self.color_correction = np.array([1., 1., 1] if color_correction is None else color_correction, dtype=np.float32)

        # set up temporary output
        self.random_id = ''.join([str(x) for x in np.random.randint(0, 9, size=10)])
        try:
            os.makedirs("/tmp/{}".format(self.random_id))
        except OSError:
            pass

        self.write_pool = multiprocessing.Pool(n_threads)
        self.n_items = 0

    def batch(self, item_num, i_rgb_rec, i_alpha_rec=None, image=None, i_rgb_sqerr=None, **kwargs):

        i_rgb_rec = i_rgb_rec.data.to("cpu").numpy().transpose((0, 2, 3, 1))
        i_alpha_rec = i_alpha_rec.data.to("cpu").numpy()[:, 0, :, :, None] if i_alpha_rec is not None else 1.0

        # color correction
        output_image = i_rgb_rec * self.color_correction[None, None, None, :]

        # composite background color
        output_image = output_image + (1. - i_alpha_rec) * self.background_color[None, None, None, :]

        # concatenate ground truth image
        if self.show_target and image is not None:
            image = image.data.to("cpu").numpy().transpose((0, 2, 3, 1))
            image = image * self.color_correction[None, None, None, :]
            output_image = np.concatenate((output_image, image), axis=2)

        # concatenate difference image
        if self.show_diff and imagediff is not None:
            i_rgb_sqerr = np.mean(i_rgb_sqerr.data.to("cpu").numpy(), axis=1)
            i_rgb_sqerr = (cm.magma(4. * i_rgb_sqerr / 255.)[:, :, :, :3] * 255.)
            output_image = np.concatenate((output_image, i_rgb_sqerr), axis=2)

        self.write_pool.map(writeimage,
                            zip([self.random_id for i in range(item_num.size(0))],
                                item_num.data.to("cpu").numpy(),
                                output_image))
        self.n_items += item_num.size(0)

    def finalize(self):
        # make video file
        command = (
            "ffmpeg -y -r 30 -i /tmp/{}/%06d.jpg "
            "-vframes {} "
            "-vcodec libx264 -crf 18 "
            "-pix_fmt yuv420p "
            "{}".format(self.random_id, self.n_items, self.outpath)
        ).split()
        subprocess.call(command)

        shutil.rmtree("/tmp/{}".format(self.random_id))
