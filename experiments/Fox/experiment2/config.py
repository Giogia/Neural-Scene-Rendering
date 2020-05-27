
import os

import src.parameters as parameters
from src.datasets.join import JoinDataset


def get_dataset(camera_list=None, frame_list=None, background=False, depth=False, subsample_type=None):
    from src.datasets.blender import Dataset
    return Dataset(
        camera_list=camera_list,
        frame_list=frame_list,
        background=background,
        depth=depth,
        fixed_cameras=["1", "3", "7"],
        image_mean=50.,
        image_std=25.,
        image_size=[960, 540],
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
        color_cal_lib.Colorcal(dataset.cameras),
        4. / 256)


# profiles
# A profile is instantiated by the training or evaluation scripts
# and controls how the dataset and autoencoder is created
class Train:
    batch_size = 8
    max_iter = 10000

    def get_autoencoder(self, dataset): return get_autoencoder(dataset)

    def get_dataset(self):
        return get_dataset(camera_list=[i+1 for i in range(parameters.CAMERAS_NUMBER)],
                           frame_list=[i for i in range(parameters.START_FRAME, parameters.END_FRAME)],
                           background=True,
                           subsample_type="random2")

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


class Progress:
    """Write out diagnostic images during training."""
    batch_size = 8

    def get_ae_args(self): return dict(output_list=["i_rgb_rec"])

    def get_dataset(self): return get_dataset(camera_list=[i+1 for i in range(parameters.CAMERAS_NUMBER)],
                                              frame_list=[parameters.END_FRAME],
                                              background=True)

    def get_writer(self):
        from src.writers.progress import ProgressWriter
        return ProgressWriter()


class Render:
    """Render model with training camera or from novel viewpoints."""

    def __init__(self, cam=None, show_target=False, view_template=False):
        self.cam = cam
        self.show_target = show_target
        self.view_template = view_template

    def get_autoencoder(self, dataset):
        return get_autoencoder(dataset)

    def get_ae_args(self):
        return dict(output_list=["i_rgb_rec", "i_alpha_rec"], view_template=self.view_template)

    def get_dataset(self):
        dataset = get_dataset(camera_list=[] if self.cam is None else [self.cam],
                              frame_list=[i for i in range(parameters.START_FRAME+50, parameters.END_FRAME)],
                              background=True)
        if self.cam is None:

            from src.datasets.rotate import Dataset
            cam_dataset = Dataset(length=len(dataset))

            return JoinDataset(cam_dataset, dataset)
        else:
            return dataset

    def get_writer(self):
        from src.writers.video import Writer
        return Writer(
            os.path.join(os.path.dirname(__file__),
                         "render_{}{}.mp4".format(
                             "rotate" if self.cam is None else self.cam,
                             "_template" if self.view_template else "")),
            show_target=self.show_target)
