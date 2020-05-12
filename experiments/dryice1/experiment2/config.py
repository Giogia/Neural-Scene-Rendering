# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import data.dryice1 as data_model


def get_dataset(camera_filter=lambda x: True, max_frames=-1, subsample_type=None):
    return data_model.Dataset(
        camera_filter=camera_filter,
        frame_list=[i for i in range(15469, 16578, 3)][:max_frames],
        key_filter=["background", "fixed_cam_image", "camera", "image", "pixel_coords"],
        fixed_cameras=["400007", "400010", "400018"],
        fixed_cam_mean=100.,
        fixed_cam_std=25.,
        image_mean=100.,
        image_std=25.,
        subsample_type=subsample_type,
        subsample_size=128,
        world_scale=1. / 256)


def get_autoencoder(dataset):
    import models.autoencoder as ae_model
    import models.encoders.mv_conv as encoder_lib
    import models.decoders.voxel as decoder_lib
    import models.volsamplers.warpvoxel as vol_sampler_lib
    import models.colorcals.color_calibrator as color_cal_lib
    return ae_model.Autoencoder(
        dataset,
        encoder_lib.Encoder(3),
        decoder_lib.Decoder(global_warp=False),
        vol_sampler_lib.VolSampler(),
        color_cal_lib.Colorcal(dataset.get_cameras()),
        4. / 256)


def get_loss():
    import models.losses.aeloss as loss
    return loss.AutoencoderLoss()


# profiles
# A profile is instantiated by the training or evaluation scripts
# and controls how the dataset and autoencoder is created
class Train:
    batch_size = 8
    max_iter = 10000

    def get_autoencoder(self, dataset): return get_autoencoder(dataset)

    def get_dataset(self): return get_dataset(subsample_type="random2")

    def get_optimizer(self, ae):
        import itertools
        import torch.optim
        lr = 0.0001
        aeparams = itertools.chain(
            [{"params": x} for x in ae.encoder.parameters()],
            [{"params": x} for x in ae.decoder.parameters()],
            [{"params": x} for x in ae.color_calibrator.parameters()])
        return torch.optim.AdamW(aeparams, lr=lr, betas=(0.9, 0.999))

    def get_loss_weights(self):
        return {"irgbmse": 1.0, "kldiv": 0.001, "alphapr": 0.01, "tvl1": 0.01}

    def get_loss(self): return get_loss()


class ProgressWriter:
    def batch(self, iter_num, **kwargs):
        import numpy as np
        from PIL import Image
        rows = []
        row = []
        for i in range(kwargs["image"].size(0)):
            row.append(
                np.concatenate((
                    kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                    kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1))
            if len(row) == 4:
                rows.append(np.concatenate(row, axis=1))
                row = []
        img_out = np.concatenate(rows, axis=0)
        outpath = os.path.dirname(__file__)
        Image.fromarray(np.clip(img_out, 0, 255).astype(np.uint8)).save(
            os.path.join(outpath, "prog_{:06}.jpg".format(iter_num)))


class Progress:
    """Write out diagnostic images during training."""
    batch_size = 8

    def get_ae_args(self): return dict(output_list=["irgbrec"])

    def get_dataset(self): return get_dataset(max_frames=1)

    def get_writer(self): return ProgressWriter()


class Render:
    """Render model with training camera or from novel viewpoints.
    
    e.g., python render.py {configpath} Render --maxframes 128"""

    def __init__(self, cam=None, max_frames=-1, show_target=False, view_template=False):
        self.cam = cam
        self.max_frames = max_frames
        self.show_target = show_target
        self.view_template = view_template

    def get_autoencoder(self, dataset):
        return get_autoencoder(dataset)

    def get_ae_args(self):
        return dict(output_list=["irgbrec"], view_template=self.view_template)

    def get_dataset(self):
        import data.utils
        import eval.cameras.rotate as cameralib
        dataset = get_dataset(camera_filter=lambda x: x == self.cam, max_frames=self.max_frames)
        if self.cam is None:
            cam_dataset = cameralib.Dataset(len(dataset))
            return data.utils.JoinDataset(cam_dataset, dataset)
        else:
            return dataset

    def get_writer(self):
        import eval.writers.videowriter as writer_lib
        return writer_lib.Writer(
            os.path.join(os.path.dirname(__file__),
                         "render_{}{}.mp4".format(
                             "rotate" if self.cam is None else self.cam,
                             "_template" if self.view_template else "")),
            showtarget=self.show_target)
