
import os


class ProgressWriter:
    def batch(self, iter_num, **kwargs):
        import numpy as np
        from PIL import Image
        rows = []
        row = []
        for i in range(kwargs["image"].size(0)):
            row.append(
                np.concatenate((
                    kwargs["i_rgb_rec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                    kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1))
            if len(row) == 2:
                rows.append(np.concatenate(row, axis=1))
                row = []
        img_out = np.concatenate(rows, axis=0)
        outpath = os.path.dirname(__file__)
        Image.fromarray(np.clip(img_out, 0, 255).astype(np.uint8)).save(
            os.path.join(outpath, "prog_{:06}.jpg".format(iter_num)))
