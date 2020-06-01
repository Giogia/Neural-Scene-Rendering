
import os

import numpy as np
import torch.utils.data

from src import parameters
from src.utils.csv import read_csv
from src.utils.exr import exr_to_image, exr_to_depth


class Dataset(torch.utils.data.Dataset):
    def __init__(self, camera_list, frame_list, background=False, depth=False,
                 fixed_cameras=[], image_mean=0., image_std=1., image_size=[0, 0],
                 world_scale=1., subsample_type=None, subsample_size=0, path=None):

        self.cameras = sorted(list(map(str, camera_list)))
        self.frame_list = frame_list
        self.frame_cam_list = [(x, cam)
                               for x in self.frame_list
                               for cam in (self.cameras if len(self.cameras) > 0 else [None])]

        self.use_background = background
        self.use_depth = depth

        self.fixed_cameras = fixed_cameras
        self.image_mean = image_mean
        self.image_std = image_std
        self.image_size = image_size
        self.subsample_type = subsample_type
        self.subsample_size = subsample_size
        self.path = path

        # compute camera positions
        self.camera_position, self.camera_rotation, self.focal, self.principal_point, self.size = {}, {}, {}, {}, {}
        intrinsic = np.array(read_csv(os.path.join(self.path, 'camera_intrinsic.csv')))
        for cam in self.cameras:
            extrinsic = np.array(read_csv(os.path.join(self.path, 'camera_' + cam, 'pose.csv')))[:][:-1]
            self.camera_position[cam] = (-np.dot(extrinsic[:3, :3].T, extrinsic[:3, 3])).astype(np.float32)
            self.camera_rotation[cam] = (extrinsic[:3, :3]).astype(np.float32)
            self.focal[cam] = np.diag(intrinsic[:2, :2]).astype(np.float32)
            self.principal_point[cam] = intrinsic[:2, 2].astype(np.float32)
            self.size[cam] = np.array(self.image_size)

        # transformation that places the center of the object at the origin
        transformation = read_csv(os.path.join(self.path, "model.csv"))
        self.model_transformation = np.array(transformation, dtype=np.float32)[0:][0:-1]
        self.model_transformation[:3, :3] *= world_scale

        # load background images for each camera
        if self.use_background:
            self.background = {}
            for cam in self.cameras:
                image_path = os.path.join(self.path, 'camera_' + cam, 'background.exr')
                image = 255 * exr_to_image(image_path).transpose((2, 0, 1)).astype(np.float32)
                self.background[cam] = image

    def __len__(self):
        return len(self.frame_cam_list)

    def __getitem__(self, index):
        frame, cam = self.frame_cam_list[index]

        result = {}

        valid_input = True

        # fixed camera images
        images = []
        for camera in self.fixed_cameras:
            image_path = os.path.join(self.path, 'camera_' + str(camera), str(frame) + '.exr')
            # sample image to half resolution
            image = 255 * exr_to_image(image_path)[::2, ::2, :].transpose((2, 0, 1)).astype(np.float32)
            if self.use_depth:
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
            # camera data
            result["camera_rotation"] = np.dot(self.model_transformation[:3, :3].T, self.camera_rotation[cam].T).T
            result["camera_position"] = np.dot(self.model_transformation[:3, :3].T,
                                               self.camera_position[cam] - self.model_transformation[:3, 3])
            result["focal"] = self.focal[cam]
            result["principal_point"] = self.principal_point[cam]
            result["camera_index"] = self.cameras.index(cam)

            # image
            image_path = os.path.join(self.path, 'camera_' + str(cam), str(frame) + '.exr')
            image = 255 * exr_to_image(image_path).transpose((2, 0, 1)).astype(np.float32)
            depth = exr_to_depth(image_path, far_threshold=2 * parameters.DISTANCE)
            depth = np.expand_dims(depth, axis=-1).transpose((2, 0, 1)).astype(np.float32)
            depth = 255 * depth / np.max(depth)
            height, width = image.shape[1:3]
            valid = np.float32(1.0) if np.sum(image) != 0 else np.float32(0.)
            result["image"] = image
            result["depth"] = depth
            result["image_valid"] = valid

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
