
import numpy as np

import torch.utils.data
from src.parameters import BACKGROUND_COLOR


class Dataset(torch.utils.data.Dataset):
    def __init__(self, length, period=128):
        self.length = length
        self.period = period
        self.width, self.height = 960, 540
        self.cameras = ['rotate']

        self.focal = np.array([1000. * (self.width / 960.), 1000. * (self.width / 960.)], dtype=np.float32)
        self.principal_point = np.array([self.width * 0.5, self.height * 0.5], dtype=np.float32)
        self.size = {self.cameras[0]: np.array([self.width, self.height])}
        self.background = {self.cameras[0]: np.stack([color * np.ones((self.height, self.width), dtype=np.float32)
                                                     for color in BACKGROUND_COLOR])}

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        t = (np.cos(index * 2. * np.pi / self.period) * 0.5 + 0.5)
        x = np.cos(t * 2 * np.pi)
        y = np.sin(t * 2 * np.pi)
        z = 0

        camera_position = np.array([x, y, z], dtype=np.float32)

        look_at = np.array([0., 0., 0.], dtype=np.float32)
        up = np.array([0., 0., -1.], dtype=np.float32)
        forward = look_at - camera_position
        forward /= np.linalg.norm(forward)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        up /= np.linalg.norm(up)

        camera_rotation = np.array([right, up, forward], dtype=np.float32)

        px, py = np.meshgrid(np.arange(self.width).astype(np.float32), np.arange(self.height).astype(np.float32))
        pixel_coords = np.stack((px, py), axis=-1)

        return {"camera_position": camera_position,
                "camera_rotation": camera_rotation,
                "focal": self.focal,
                "principal_point": self.principal_point,
                "pixel_coords": pixel_coords}
