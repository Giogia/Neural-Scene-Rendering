
import torch
import torch.nn as nn

import models.utils


class Encoder(torch.nn.Module):

    def __init__(self, n_inputs, n_channels=3, tied=False):
        super(Encoder, self).__init__()

        self.n_inputs = n_inputs
        self.n_channels = n_channels
        self.tied = tied

        self.down1 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(n_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, 4, 2, 1), nn.LeakyReLU(0.2))
            for i in range(1 if self.tied else self.n_inputs)])

        self.down2 = nn.Sequential(
            nn.Linear(256 * self.n_inputs * 2 * 4, 512), nn.LeakyReLU(0.2))

        height, width = 540, 960
        y_pad = ((height + 127) // 128) * 128 - height
        x_pad = ((width + 127) // 128) * 128 - width

        self.pad = nn.ZeroPad2d((x_pad // 2, x_pad - x_pad // 2, y_pad // 2, y_pad - y_pad // 2))
        self.mu = nn.Linear(512, 256)
        self.log_std = nn.Linear(512, 256)

        for i in range(1 if self.tied else self.n_inputs):
            models.utils.initseq(self.down1[i])

        models.utils.initseq(self.down2)
        models.utils.initmod(self.mu)
        models.utils.initmod(self.log_std)

    def forward(self, x, loss_list=[]):

        x = self.pad(x)
        x = [self.down1[0 if self.tied else i](x[:, i * self.n_channels:(i + 1) * self.n_channels, :, :])
                 .view(-1, 256 * 2 * 4) for i in range(self.n_inputs)]
        x = torch.cat(x, dim=1)
        x = self.down2(x)

        mu, log_std = self.mu(x) * 0.1, self.log_std(x) * 0.01

        if self.training:
            z = mu + torch.exp(log_std) * torch.randn(*log_std.size(), device=log_std.device)
        else:
            z = mu

        losses = {}
        if "kl_div" in loss_list:
            losses["kl_div"] = torch.mean(-0.5 - log_std + 0.5 * mu ** 2 + 0.5 * torch.exp(2 * log_std), dim=-1)

        return {"encoding": z, "losses": losses}