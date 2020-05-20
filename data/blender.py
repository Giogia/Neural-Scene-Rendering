# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
import torch.utils.data
import os

from .csv_utils import read_csv
from .exr_utils import exr_to_image, exr_to_depth
from . import parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_cameras(camera_list, path):

    cameras = {}
    intrinsic = read_csv(os.path.join(path, 'camera_intrinsic.csv'))

    for camera_name in camera_list:
        name = str(camera_name)
        extrinsic = read_csv(os.path.join(path, 'camera_' + name, 'pose.csv'))
        cameras[name] = {
            "intrinsic": np.array(intrinsic),
            "dist": np.array([0 for i in range(5)]),
            "extrinsic": np.array(extrinsic)[:][:-1]
        }

    return cameras


class Dataset(torch.utils.data.Dataset):
    def __init__(self, camera_filter, camera_list, frame_list, key_filter,
                 fixed_cameras=[], image_mean=0., image_std=1.,
                 world_scale=1., subsample_type=None, subsample_size=0, path=None):

        self.path = path
        cameras = load_cameras(camera_list, self.path)

        # get options
        self.cameras = sorted(list(filter(camera_filter, cameras.keys())))
        self.frame_list = frame_list
        self.frame_cam_list = [(x, cam)
                               for x in self.frame_list
                               for cam in (self.cameras if len(self.cameras) > 0 else [None])]

        self.key_filter = key_filter
        self.fixed_cameras = fixed_cameras
        self.image_mean = image_mean
        self.image_std = image_std
        self.subsample_type = subsample_type
        self.subsample_size = subsample_size

        # compute camera positions
        self.campos, self.cam_rot, self.focal, self.princ_pt = {}, {}, {}, {}
        for cam in self.cameras:
            self.campos[cam] = (-np.dot(cameras[cam]['extrinsic'][:3, :3].T, cameras[cam]['extrinsic'][:3, 3])).astype(
                np.float32)
            self.cam_rot[cam] = (cameras[cam]['extrinsic'][:3, :3]).astype(np.float32)
            self.focal[cam] = np.diag(cameras[cam]['intrinsic'][:2, :2]).astype(np.float32)
            self.princ_pt[cam] = cameras[cam]['intrinsic'][:2, 2].astype(np.float32)

        # transformation that places the center of the object at the origin
        transformation = read_csv(os.path.join(self.path, "model.csv"))
        self.model_transformation = np.array(transformation, dtype=np.float32)[0:][0:-1]
        self.model_transformation[:3, :3] *= world_scale

        # load background images for each camera
        if "background" in self.key_filter:
            self.background = {}
            for camera_name in self.cameras:
                try:
                    image_path = os.path.join(self.path, 'camera_' + camera_name, 'background.exr')
                    image = 255 * exr_to_image(image_path).transpose((2, 0, 1)).astype(np.float32)
                    self.background[camera_name] = image
                except KeyboardInterrupt:
                    pass

    def get_cameras(self):
        return self.cameras

    def get_krt(self):
        return {k: {
            "pos": self.campos[k],
            "rot": self.cam_rot[k],
            "focal": self.focal[k],
            "princpt": self.princ_pt[k],
            "size": np.array([960, 540])}
            for k in self.cameras}

    def known_background(self):
        return "background" in self.key_filter

    def get_background(self, background):
        if "background" in self.key_filter:
            for camera_name in self.cameras:
                if camera_name in self.background.keys():
                    background[camera_name].data[:] = torch.from_numpy(self.background[camera_name]).to(device)

    def __len__(self):
        return len(self.frame_cam_list)

    def __getitem__(self, idx):
        frame, cam = self.frame_cam_list[idx]

        result = {}

        valid_input = True

        # fixed camera images
        if "fixed_cam_image" in self.key_filter:
            n_input = len(self.fixed_cameras)

            images = []
            for i in range(n_input):
                image_path = os.path.join(self.path, 'camera_' + str(self.fixed_cameras[i]), str(frame) + '.exr')
                image = 255 * exr_to_image(image_path)[::2, ::2, :].transpose((2, 0, 1)).astype(np.float32)
                if "depth" in self.key_filter:
                    depth = exr_to_depth(image_path, far_threshold=2 * parameters.DISTANCE)
                    depth = np.expand_dims(depth, axis=-1)[::2, ::2, :].transpose((2, 0, 1)).astype(np.float32)
                    depth = 255 * depth / np.max(depth)
                    image = np.append(image, depth, axis=0)
                if np.sum(image) == 0:
                    valid_input = False
                images.append(image)

            fixed_cam_image = np.concatenate(([image for image in images]), axis=0)

            fixed_cam_image[:] -= self.image_mean
            fixed_cam_image[:] /= self.image_std
            result["fixed_cam_image"] = fixed_cam_image

        result["valid_input"] = np.float32(1.0 if valid_input else 0.0)

        # image data
        if cam is not None:
            if "camera" in self.key_filter:
                # camera data
                result["camera_rotation"] = np.dot(self.model_transformation[:3, :3].T, self.cam_rot[cam].T).T
                result["camera_position"] = np.dot(self.model_transformation[:3, :3].T,
                                                   self.campos[cam] - self.model_transformation[:3, 3])
                result["focal"] = self.focal[cam]
                result["principal_point"] = self.princ_pt[cam]
                result["camera_index"] = self.cameras.index(cam)

            if "image" in self.key_filter:
                # image
                image_path = os.path.join(self.path, 'camera_' + str(cam), str(frame) + '.exr')
                image = 255 * exr_to_image(image_path).transpose((2, 0, 1)).astype(np.float32)
                height, width = image.shape[1:3]
                valid = np.float32(1.0) if np.sum(image) != 0 else np.float32(0.)
                result["image"] = image
                result["image_valid"] = valid

            if "pixel_coords" in self.key_filter:
                if self.subsample_type == "patch":
                    ind_x = np.random.randint(0, width - self.subsample_size + 1)
                    ind_y = np.random.randint(0, height - self.subsample_size + 1)

                    px, py = np.meshgrid(
                        np.arange(ind_x, ind_x + self.subsample_size).astype(np.float32),
                        np.arange(ind_y, ind_y + self.subsample_size).astype(np.float32))
                elif self.subsample_type == "random":
                    px = np.random.randint(0, width, size=(self.subsample_size, self.subsample_size)).astype(np.float32)
                    py = np.random.randint(0, height, size=(self.subsample_size, self.subsample_size)).astype(
                        np.float32)
                elif self.subsample_type == "random2":
                    px = np.random.uniform(0, width - 1e-5, size=(self.subsample_size, self.subsample_size)).astype(
                        np.float32)
                    py = np.random.uniform(0, height - 1e-5, size=(self.subsample_size, self.subsample_size)).astype(
                        np.float32)
                else:
                    px, py = np.meshgrid(np.arange(width).astype(np.float32), np.arange(height).astype(np.float32))

                result["pixel_coords"] = np.stack((px, py), axis=-1)

        return result
