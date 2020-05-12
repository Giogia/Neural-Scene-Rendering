import torch
import torch.nn as nn


class Colorcal(nn.Module):
    """Apply learnable 3 channel scale and bias to an image to handle un(color)calibrated cameras."""

    def __init__(self, cameras):
        super(Colorcal, self).__init__()

        self.cameras = cameras

        self.conv = nn.ModuleDict({
            k: nn.Conv2d(3, 3, 1, 1, 0, groups=3) for k in self.cameras})

        for k in self.cameras:
            self.conv[k].weight.data[:] = 1.
            self.conv[k].bias.data.zero_()

    def forward(self, image, camera_index):
        return torch.cat(
            [self.conv[self.cameras[camera_index[i].item()]](image[i:i + 1, :, :, :]) for i in range(image.size(0))])
