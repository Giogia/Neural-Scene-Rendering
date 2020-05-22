
import os

import data.blender as data_model
import data.parameters as parameters


def get_dataset(camera_filter=lambda x: True, frame_list=None, subsample_type=None):
    if frame_list is None:
        frame_list = [i for i in range(parameters.START_FRAME, parameters.END_FRAME)]
    return data_model.Dataset(
        camera_filter=camera_filter,
        camera_list=[i+1 for i in range(parameters.CAMERAS_NUMBER)],
        frame_list=frame_list,
        key_filter=["background", "fixed_cam_image", "camera", "image", "pixel_coords"],
        fixed_cameras=["1", "3", "7"],
        image_mean=50.,
        image_std=25.,
        subsample_type=subsample_type,
        subsample_size=128,
        world_scale=parameters.SCALE,
        path=os.path.join('experiments', 'Fox', 'data'))


def get_autoencoder(dataset):
    import models.autoencoder as ae_model
    import models.encoders.conv as encoder_lib
    import models.decoders.voxel as decoder_lib
    import models.volsamplers.warpvoxel as vol_sampler_lib
    import models.colorcals.color_calibrator as color_cal_lib
    return ae_model.Autoencoder(
        dataset,
        encoder_lib.Encoder(n_inputs=3, n_channels=3),
        decoder_lib.Decoder(global_warp=False),
        vol_sampler_lib.VolSampler(),
        color_cal_lib.Colorcal(dataset.get_cameras()),
        4. / 256)


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
        ae_params = itertools.chain(
            [{"params": x} for x in ae.encoder.parameters()],
            [{"params": x} for x in ae.decoder.parameters()],
            [{"params": x} for x in ae.color_calibrator.parameters()])
        return torch.optim.AdamW(ae_params, lr=lr, betas=(0.9, 0.999))

    def get_loss_weights(self):
        return {"i_rgb_mse": 1.0, "kl_div": 0.001, "alpha_prior": 0.01, "tvl1": 0.01}

    def get_loss(self):
        import models.losses.aeloss as loss
        return loss.AutoencoderLoss()

    def get_scheduler(self, optimizer, base_lr, max_lr, iter_num, step=100):
        from torch.optim.lr_scheduler import CyclicLR
        return CyclicLR(optimizer, base_lr, max_lr,
                        step_size_up=step,
                        cycle_momentum=False,
                        last_epoch=iter_num - 1)


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


class Progress:
    """Write out diagnostic images during training."""
    batch_size = 8

    def get_ae_args(self): return dict(output_list=["i_rgb_rec"])

    def get_dataset(self): return get_dataset(frame_list=[parameters.END_FRAME])

    def get_writer(self): return ProgressWriter()


class Render:
    """Render model with training camera or from novel viewpoints.

    e.g., python render.py {configpath} Render --maxframes 128"""

    def __init__(self, cam=None, show_target=True, view_template=False):
        self.cam = cam
        self.show_target = show_target
        self.view_template = view_template

    def get_autoencoder(self, dataset):
        return get_autoencoder(dataset)

    def get_ae_args(self):
        return dict(output_list=["i_rgb_rec", "i_alpha_rec"], view_template=self.view_template)

    def get_dataset(self):
        import data.utils
        import eval.cameras.rotate as cameralib
        dataset = get_dataset(camera_filter=lambda x: x == self.cam)
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
            show_target=self.show_target)
