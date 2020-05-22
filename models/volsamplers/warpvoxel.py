
import torch
import torch.nn as nn
import torch.nn.functional as F


class VolSampler(nn.Module):
    def __init__(self, displacement_warp=False):
        super(VolSampler, self).__init__()

        self.displacement_warp = displacement_warp

    def forward(self, pos, template, warp=None, g_warp_s=None, g_warp_rot=None, g_warp_t=None, view_template=False, **kwargs):
        valid = None
        if not view_template:
            if g_warp_s is not None:
                pos = (torch.sum(
                    (pos - g_warp_t[:, None, None, None, :])[:, :, :, :, None, :] *
                    g_warp_rot[:, None, None, None, :, :], dim=-1) *
                       g_warp_s[:, None, None, None, :])
            if warp is not None:
                if self.displacement_warp:
                    pos = pos + F.grid_sample(warp, pos).permute(0, 2, 3, 4, 1)
                else:
                    valid = torch.prod((pos > -1.) * (pos < 1.), dim=-1).float()
                    pos = F.grid_sample(warp, pos).permute(0, 2, 3, 4, 1)
        val = F.grid_sample(template, pos)
        if valid is not None:
            val = val * valid[:, None, :, :, :]
        return val[:, :3, :, :, :], val[:, 3:, :, :, :]
