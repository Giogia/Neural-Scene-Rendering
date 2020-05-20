# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np

import torch
import torch.utils.data
from data.parameters import DISTANCE

class Dataset(torch.utils.data.Dataset):
    def __init__(self, length, period=128):
        self.length = length
        self.period = period
        self.width, self.height = 480, 640

        self.focal = np.array([1000. * (self.width / 480.), 1000. * (self.width / 480.)], dtype=np.float32)
        self.principal_point = np.array([self.width * 0.5, self.height * 0.5], dtype=np.float32)

    def __len__(self):
        return self.length

    def get_cameras(self):
        return ["rotate"]

    def get_krt(self):
        return {"rotate": {
            "focal": self.focal,
            "principal_point": self.principal_point,
            "size": np.array([self.width, self.height])}}

    def __getitem__(self, idx):

        t = (np.cos(idx * 2. * np.pi / self.period) * 0.5 + 0.5)
        x = np.cos(t * 0.5 * np.pi + 0.25 * np.pi) * DISTANCE
        y = np.sin(t * 0.5 * np.pi + 0.25 * np.pi) * DISTANCE
        z = 0

        camera_position = np.array([x, y, z], dtype=np.float32)

        look_at = np.array([0., 0., 0.], dtype=np.float32)
        up = np.array([0., -1., 0.], dtype=np.float32)
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
