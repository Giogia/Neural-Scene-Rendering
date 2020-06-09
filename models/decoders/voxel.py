
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.utils


class ConvTemplate(nn.Module):
    def __init__(self, encoding_size=256, out_channels=4, template_res=128):
        super(ConvTemplate, self).__init__()

        self.encoding_size = encoding_size
        self.out_channels = out_channels
        self.template_res = template_res

        # build template convolution stack
        self.template1 = nn.Sequential(nn.Linear(self.encoding_size, 1024), nn.LeakyReLU(0.2))
        template2 = []
        in_channels, out_channels = 1024, 512
        for i in range(int(np.log2(self.template_res)) - 1):
            template2.append(nn.ConvTranspose3d(in_channels, out_channels, 4, 2, 1))
            template2.append(nn.LeakyReLU(0.2))
            if in_channels == out_channels:
                out_channels = in_channels // 2
            else:
                in_channels = out_channels
        template2.append(nn.ConvTranspose3d(in_channels, 4, 4, 2, 1))
        self.template2 = nn.Sequential(*template2)

        for m in [self.template1, self.template2]:
            models.utils.initseq(m)

    def forward(self, encoding):
        return self.template2(self.template1(encoding).view(-1, 1024, 1, 1, 1))


class LinearTemplate(nn.Module):
    def __init__(self, encoding_size=256, out_channels=4, template_res=128):
        super(LinearTemplate, self).__init__()

        self.encoding_size = encoding_size
        self.out_channels = out_channels
        self.template_res = template_res

        self.template1 = nn.Sequential(
            nn.Linear(self.encoding_size, 8), nn.LeakyReLU(0.2),
            nn.Linear(8, self.template_res ** 3 * self.out_channels))

        for m in [self.template1]:
            models.utils.initseq(m)

    def forward(self, encoding):
        return self.template1(encoding).view(-1, self.out_channels, self.template_res, self.template_res, self.template_res)


def get_template(template_type, **kwargs):
    if template_type == "conv":
        return ConvTemplate(**kwargs)
    elif template_type == "affine_mix":
        return LinearTemplate(**kwargs)
    else:
        return None


class ConvWarp(nn.Module):
    def __init__(self, displacement_warp=False, **kwargs):
        super(ConvWarp, self).__init__()

        self.displacement_warp = displacement_warp

        self.warp1 = nn.Sequential(
            nn.Linear(256, 1024), nn.LeakyReLU(0.2))
        self.warp2 = nn.Sequential(
            nn.ConvTranspose3d(1024, 512, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(512, 512, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(512, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.ConvTranspose3d(256, 3, 4, 2, 1))
        for m in [self.warp1, self.warp2]:
            models.utils.initseq(m)

        z_grid, y_grid, x_grid = np.meshgrid(
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1.0, 1.0, 32), indexing='ij')
        self.register_buffer("grid", torch.tensor(np.stack((x_grid, y_grid, z_grid), axis=0)[None].astype(np.float32)))

    def forward(self, encoding):
        final_warp = self.warp2(self.warp1(encoding).view(-1, 1024, 1, 1, 1)) * (2. / 1024)
        if not self.displacement_warp:
            final_warp = final_warp + self.grid
        return final_warp


class AffineMixWarp(nn.Module):
    def __init__(self, **kwargs):
        super(AffineMixWarp, self).__init__()

        self.quaternion = models.utils.Quaternion()

        self.warp_s = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 3 * 16))
        self.warp_r = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 4 * 16))
        self.warp_t = nn.Sequential(
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 3 * 16))
        self.weight_branch = nn.Sequential(
            nn.Linear(256, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 16 * 32 * 32 * 32))
        for m in [self.warp_s, self.warp_r, self.warp_t, self.weight_branch]:
            models.utils.initseq(m)

        z_grid, y_grid, x_grid = np.meshgrid(
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1.0, 1.0, 32),
            np.linspace(-1.0, 1.0, 32), indexing='ij')
        self.register_buffer("grid", torch.tensor(np.stack((x_grid, y_grid, z_grid), axis=-1)[None].astype(np.float32)))

    def forward(self, encoding):
        warp_s = self.warp_s(encoding).view(encoding.size(0), 16, 3)
        warp_r = self.warp_r(encoding).view(encoding.size(0), 16, 4)
        warp_t = self.warp_t(encoding).view(encoding.size(0), 16, 3) * 0.1
        warp_rot = self.quaternion(warp_r.view(-1, 4)).view(encoding.size(0), 16, 3, 3)

        weight = torch.exp(self.weight_branch(encoding).view(encoding.size(0), 16, 32, 32, 32))

        warped_weight = torch.cat([
            F.grid_sample(weight[:, i:i + 1, :, :, :],
                          torch.sum(((self.grid - warp_t[:, None, None, None, i, :])[:, :, :, :, None, :] *
                                     warp_rot[:, None, None, None, i, :, :]), dim=5) *
                          warp_s[:, None, None, None, i, :], padding_mode='border')
            for i in range(weight.size(1))], dim=1)

        warp = torch.sum(torch.stack([
            warped_weight[:, i, :, :, :, None] *
            (torch.sum(((self.grid - warp_t[:, None, None, None, i, :])[:, :, :, :, None, :] *
                        warp_rot[:, None, None, None, i, :, :]), dim=5) *
             warp_s[:, None, None, None, i, :])
            for i in range(weight.size(1))], dim=1), dim=1) / torch.sum(warped_weight, dim=1).clamp(min=0.001)[:, :, :,
                                                              :, None]

        return warp.permute(0, 4, 1, 2, 3)


def get_warp(warptype, **kwargs):
    if warptype == "conv":
        return ConvWarp(**kwargs)
    elif warptype == "affine_mix":
        return AffineMixWarp(**kwargs)
    else:
        return None


class Decoder(nn.Module):
    def __init__(self, template_type="conv", template_res=128,
                 view_conditioned=False, global_warp=True, warp_type="affine_mix",
                 displacement_warp=False):
        super(Decoder, self).__init__()

        self.template_type = template_type
        self.template_res = template_res
        self.view_conditioned = view_conditioned
        self.global_warp = global_warp
        self.warp_type = warp_type
        self.displacement_warp = displacement_warp

        if self.view_conditioned:
            self.template = get_template(self.template_type, encoding_size=256 + 3,
                                         out_channels=3, template_res=self.template_res)
            self.template_alpha = get_template(self.template_type, encoding_size=256,
                                               out_channels=1, template_res=self.template_res)
        else:
            self.template = get_template(self.template_type, template_res=self.template_res)

        self.warp = get_warp(self.warp_type, displacement_warp=self.displacement_warp)

        if self.global_warp:
            self.quaternion = models.utils.Quaternion()

            self.g_warp_s = nn.Sequential(
                nn.Linear(256, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 3))
            self.g_warp_r = nn.Sequential(
                nn.Linear(256, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 4))
            self.g_warp_t = nn.Sequential(
                nn.Linear(256, 128), nn.LeakyReLU(0.2),
                nn.Linear(128, 3))

            init_seq = models.utils.initseq
            for m in [self.g_warp_s, self.g_warp_r, self.g_warp_t]:
                init_seq(m)

    def forward(self, encoding, view_pos, loss_list=[], image_mean=0., image_std=1.):
        scale = torch.tensor([image_std, image_std, image_std, 1.], device=encoding.device)[None, :, None, None, None]
        bias = torch.tensor([image_mean, image_mean, image_mean, 0.], device=encoding.device)[None, :, None, None, None]

        # run template branch
        view_dir = view_pos / torch.sqrt(torch.sum(view_pos ** 2, dim=-1, keepdim=True))
        template_in = torch.cat([encoding, view_dir], dim=1) if self.view_conditioned else encoding
        template = self.template(template_in)
        if self.view_conditioned:
            # run alpha branch without viewpoint information
            template = torch.cat([template, self.template_alpha(encoding)], dim=1)
        # scale up to 0-255 range approximately
        template = F.softplus(bias + scale * template)

        # compute warp voxel field
        warp = self.warp(encoding) if self.warp is not None else None

        if self.global_warp:
            # compute single affine transformation
            g_warp_s = 1.0 * torch.exp(0.05 * self.g_warp_s(encoding).view(encoding.size(0), 3))
            g_warp_r = self.g_warp_r(encoding).view(encoding.size(0), 4) * 0.1
            g_warp_t = self.g_warp_t(encoding).view(encoding.size(0), 3) * 0.025
            g_warp_rot = self.quaternion(g_warp_r.view(-1, 4)).view(encoding.size(0), 3, 3)

        losses = {}

        # tv-L1 prior
        if "tvl1" in loss_list:
            log_alpha = torch.log(1e-5 + template[:, -1, :, :, :])
            losses["tvl1"] = torch.mean(torch.sqrt(1e-5 +
                                                   (log_alpha[:, :-1, :-1, 1:] - log_alpha[:, :-1, :-1, :-1]) ** 2 +
                                                   (log_alpha[:, :-1, 1:, :-1] - log_alpha[:, :-1, :-1, :-1]) ** 2 +
                                                   (log_alpha[:, 1:, :-1, :-1] - log_alpha[:, :-1, :-1, :-1]) ** 2))

        return {"template": template, "warp": warp,
                **({"g_warp_s": g_warp_s, "g_warp_rot": g_warp_rot, "g_warp_t": g_warp_t} if self.global_warp else {}),
                "losses": losses}
