import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, dataset, encoder, decoder, volume_sampler, color_calibrator, dt, step_jitter=0.01,
                 estimate_background=False):
        super(Autoencoder, self).__init__()

        self.estimate_background = estimate_background
        self.cameras = dataset.get_cameras()

        self.encoder = encoder
        self.decoder = decoder
        self.volume_sampler = volume_sampler

        self.background = nn.ParameterDict({
            k: nn.Parameter(torch.ones(3, v["size"][1], v["size"][0]), requires_grad=estimate_background)
            for k, v in dataset.get_krt().items()})

        self.color_calibrator = color_calibrator
        self.dt = dt
        self.step_jitter = step_jitter

        self.image_mean = dataset.image_mean
        self.image_std = dataset.image_std

        if dataset.known_background():
            dataset.get_background(self.background)

    # omit background from state_dict if it's not being estimated
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        ret = super(Autoencoder, self).state_dict(destination, prefix, keep_vars)
        if not self.estimate_background:
            for k in self.background.keys():
                del ret[prefix + "background." + k]
        return ret

    def forward(self, loss_list, camera_rotation, camera_position, focal, principal_point, pixel_coords, valid_input,
                fixed_cam_image=None, encoding=None, keypoints=None, camera_index=None,
                image=None, image_valid=None, view_template=False,
                output_list=[]):

        result = {"losses": {}}

        # encode input or get encoding
        if encoding is None:
            encoder_output = self.encoder(fixed_cam_image, loss_list)
            encoding = encoder_output["encoding"]
            result["losses"].update(encoder_output["losses"])

        # decode
        decoder_output = self.decoder(encoding, camera_position, loss_list)
        result["losses"].update(decoder_output["losses"])

        # NHWC
        ray_direction = (pixel_coords - principal_point[:, None, None, :]) / focal[:, None, None, :]
        ray_direction = torch.cat([ray_direction, torch.ones_like(ray_direction[:, :, :, 0:1])], dim=-1)
        ray_direction = torch.sum(camera_rotation[:, None, None, :, :] * ray_direction[:, :, :, :, None], dim=-2)
        ray_direction = ray_direction / torch.sqrt(torch.sum(ray_direction ** 2, dim=-1, keepdim=True))

        # compute raymarching starting points
        with torch.no_grad():
            t1 = (-1.0 - camera_position[:, None, None, :]) / ray_direction
            t2 = (1.0 - camera_position[:, None, None, :]) / ray_direction
            t_min = torch.max(torch.min(t1[..., 0], t2[..., 0]),
                             torch.max(torch.min(t1[..., 1], t2[..., 1]),
                                       torch.min(t1[..., 2], t2[..., 2])))
            t_max = torch.min(torch.max(t1[..., 0], t2[..., 0]),
                             torch.min(torch.max(t1[..., 1], t2[..., 1]),
                                       torch.max(t1[..., 2], t2[..., 2])))

            intersections = t_min < t_max
            t = torch.where(intersections, t_min, torch.zeros_like(t_min)).clamp(min=0.)
            t_min = torch.where(intersections, t_min, torch.zeros_like(t_min))
            t_max = torch.where(intersections, t_max, torch.zeros_like(t_min))

        # random starting point
        t = t - self.dt * torch.rand_like(t)

        ray_pos = camera_position[:, None, None, :] + ray_direction * t[..., None]  # NHWC
        ray_rgb = torch.zeros_like(ray_pos.permute(0, 3, 1, 2))  # NCHW
        ray_alpha = torch.zeros_like(ray_rgb[:, 0:1, :, :])  # NCHW

        # raymarch
        done = torch.zeros_like(t).bool()
        while not done.all():
            valid = torch.prod(torch.gt(ray_pos, -1.0) * torch.lt(ray_pos, 1.0), dim=-1).byte()
            valid_f = valid.float()

            sample_rgb, sample_alpha = self.volume_sampler(ray_pos[:, None, :, :, :], **decoder_output, viewtemplate=view_template)

            with torch.no_grad():
                step = self.dt * torch.exp(self.step_jitter * torch.randn_like(t))
                done = done | ((t + step) >= t_max)

            contrib = ((ray_alpha + sample_alpha[:, :, 0, :, :] * step[:, None, :, :]).clamp(
                max=1.) - ray_alpha) * valid_f[:, None, :, :]

            ray_rgb = ray_rgb + sample_rgb[:, :, 0, :, :] * contrib
            ray_alpha = ray_alpha + contrib

            ray_pos = ray_pos + ray_direction * step[:, :, :, None]
            t = t + step

        if image is not None:
            image_size = torch.tensor(image.size()[3:1:-1], dtype=torch.float32, device=pixel_coords.device)
            sample_coords = pixel_coords * 2. / (image_size[None, None, None, :] - 1.) - 1.

        # color correction / background
        if camera_index is not None:
            ray_rgb = self.color_calibrator(ray_rgb, camera_index)

            if pixel_coords.size()[1:3] != image.size()[2:4]:
                background = F.grid_sample(
                    torch.stack([self.background[self.cameras[camera_index[i].item()]] for i in range(camera_position.size(0))], dim=0),
                    sample_coords)
            else:
                background = torch.stack([self.background[self.cameras[camera_index[i].item()]] for i in range(camera_position.size(0))], dim=0)

            ray_rgb = ray_rgb + (1. - ray_alpha) * background.clamp(min=0.)

        if "irgbrec" in output_list:
            result["irgbrec"] = ray_rgb
        if "ialpharec" in output_list:
            result["ialpharec"] = ray_alpha

        # opacity prior
        if "alphapr" in loss_list:
            alphaprior = torch.mean(
                torch.log(0.1 + ray_alpha.view(ray_alpha.size(0), -1)) +
                torch.log(0.1 + 1. - ray_alpha.view(ray_alpha.size(0), -1)) - -2.20727, dim=-1)
            result["losses"]["alphapr"] = alphaprior

        # irgb loss
        if image is not None:
            if pixel_coords.size()[1:3] != image.size()[2:4]:
                image = F.grid_sample(image, sample_coords, align_corners=True)

            # standardize
            ray_rgb = (ray_rgb - self.image_mean) / self.image_std
            image = (image - self.image_mean) / self.image_std

            # compute reconstruction loss weighting
            if image_valid is not None:
                weight = image_valid[:, None, None, None].expand_as(image) * valid_input[:, None, None, None]
            else:
                weight = torch.ones_like(image) * valid_input[:, None, None, None]

            irgbsqerr = weight * (image - ray_rgb) ** 2

            if "irgbsqerr" in output_list:
                result["irgbsqerr"] = irgbsqerr

            if "irgbmse" in loss_list:
                irgbmse = torch.sum(irgbsqerr.view(irgbsqerr.size(0), -1), dim=-1)
                irgbmse_weight = torch.sum(weight.view(weight.size(0), -1), dim=-1)

                result["losses"]["irgbmse"] = (irgbmse, irgbmse_weight)

        return result
