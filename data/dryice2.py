# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import numpy as np
from PIL import Image
import torch.utils.data

from .csv_utils import read_csv
import parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_krt(path):
    """Load KRT file containing intrinsic and extrinsic parameters."""
    cameras = {}

    with open(path, "r") as f:

        for i in range(parameters.CAMERAS_NUMBER):
            name = str(i) + 1

            instrinsic = read_csv(file)

            intrinsic = [[float(x) for x in f.readline().split()] for i in range(3)]
            dist = [float(x) for x in f.readline().split()]
            extrinsic = [[float(x) for x in f.readline().split()] for i in range(3)]
            f.readline()

            cameras[name[:-1]] = {
                "intrinsic": np.array(intrinsic),
                "dist": np.array(dist),
                "extrinsic": np.array(extrinsic)}

    return cameras

krt_path = "../experiments/dryice1/data/KRT"
krt = load_krt(krt_path)
print(krt)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, camera_filter, frame_list, key_filter,
                 fixed_cameras=[], fixed_cam_mean=0., fixed_cam_std=1.,
                 image_mean=0., image_std=1.,
                 world_scale=1., subsample_type=None, subsample_size=0):
        krt_path = "experiments/dryice1/data/KRT"
        krt = load_krt(krt_path)

        # get options
        self.all_cameras = sorted(list(krt.keys()))
        self.cameras = list(filter(camera_filter, self.all_cameras))
        self.frame_list = frame_list
        self.frame_cam_list = [(x, cam)
                               for x in self.frame_list
                               for cam in (self.cameras if len(self.cameras) > 0 else [None])]

        self.key_filter = key_filter
        self.fixed_cameras = fixed_cameras
        self.fixed_cam_mean = fixed_cam_mean
        self.fixed_cam_std = fixed_cam_std
        self.image_mean = image_mean
        self.image_std = image_std
        self.subsample_type = subsample_type
        self.subsample_size = subsample_size

        # compute camera positions
        self.campos, self.cam_rot, self.focal, self.princ_pt = {}, {}, {}, {}
        for cam in self.cameras:
            self.campos[cam] = (-np.dot(krt[cam]['extrinsic'][:3, :3].T, krt[cam]['extrinsic'][:3, 3])).astype(np.float32)
            self.cam_rot[cam] = (krt[cam]['extrinsic'][:3, :3]).astype(np.float32)
            self.focal[cam] = (np.diag(krt[cam]['intrinsic'][:2, :2]) / 4.).astype(np.float32)
            self.princ_pt[cam] = (krt[cam]['intrinsic'][:2, 2] / 4.).astype(np.float32)

        # transformation that places the center of the object at the origin
        transf_path = "experiments/dryice1/data/pose.txt"
        self.transf = np.genfromtxt(transf_path, dtype=np.float32, skip_footer=2)
        self.transf[:3, :3] *= world_scale

        # load background images for each camera
        if "bg" in self.key_filter:
            self.bg = {}
            for i, cam in enumerate(self.cameras):
                try:
                    image_path = "experiments/dryice1/data/cam{}/bg.jpg".format(cam)
                    image = np.asarray(Image.open(image_path), dtype=np.uint8).transpose((2, 0, 1)).astype(np.float32)
                    self.bg[cam] = image
                except:
                    pass

    def get_allcameras(self):
        return self.all_cameras

    def get_krt(self):
        return {k: {
            "pos": self.campos[k],
            "rot": self.cam_rot[k],
            "focal": self.focal[k],
            "princpt": self.princ_pt[k],
            "size": np.array([667, 1024])}
            for k in self.cameras}

    def known_background(self):
        return "bg" in self.key_filter

    def get_background(self, bg):
        if "bg" in self.key_filter:
            for i, cam in enumerate(self.cameras):
                if cam in self.bg:
                    bg[cam].data[:] = torch.from_numpy(self.bg[cam]).to(device)

    def __len__(self):
        return len(self.frame_cam_list)

    def __getitem__(self, idx):
        frame, cam = self.frame_cam_list[idx]

        result = {}

        valid_input = True

        # fixed camera images
        if "fixedcamimage" in self.key_filter:
            n_input = len(self.fixed_cameras)

            fixed_cam_image = np.zeros((3 * n_input, 512, 334), dtype=np.float32)
            for i in range(n_input):
                image_path = ("experiments/dryice1/data/cam{}/image{:04}.jpg".format(self.fixed_cameras[i], int(frame)))
                image = np.asarray(Image.open(image_path), dtype=np.uint8)[::2, ::2, :].transpose((2, 0, 1)).astype(
                    np.float32)
                if np.sum(image) == 0:
                    valid_input = False
                fixed_cam_image[i * 3:(i + 1) * 3, :, :] = image
            fixed_cam_image[:] -= self.image_mean
            fixed_cam_image[:] /= self.image_std
            result["fixedcamimage"] = fixed_cam_image

        result["validinput"] = np.float32(1.0 if valid_input else 0.0)

        # image data
        if cam is not None:
            if "camera" in self.key_filter:
                # camera data
                result["camrot"] = np.dot(self.transf[:3, :3].T, self.cam_rot[cam].T).T
                result["campos"] = np.dot(self.transf[:3, :3].T, self.campos[cam] - self.transf[:3, 3])
                result["focal"] = self.focal[cam]
                result["princpt"] = self.princ_pt[cam]
                result["camindex"] = self.all_cameras.index(cam)

            if "image" in self.key_filter:
                # image
                image_path = ("experiments/dryice1/data/cam{}/image{:04}.jpg".format(cam, int(frame)))
                image = np.asarray(Image.open(image_path), dtype=np.uint8).transpose((2, 0, 1)).astype(np.float32)
                height, width = image.shape[1:3]
                valid = np.float32(1.0) if np.sum(image) != 0 else np.float32(0.)
                result["image"] = image
                result["imagevalid"] = valid

            if "pixelcoords" in self.key_filter:
                if self.subsample_type == "patch":
                    ind_x = np.random.randint(0, width - self.subsample_size + 1)
                    ind_y = np.random.randint(0, height - self.subsample_size + 1)

                    px, py = np.meshgrid(
                        np.arange(ind_x, ind_x + self.subsample_size).astype(np.float32),
                        np.arange(ind_y, ind_y + self.subsample_size).astype(np.float32))
                elif self.subsample_type == "random":
                    px = np.random.randint(0, width, size=(self.subsample_size, self.subsample_size)).astype(np.float32)
                    py = np.random.randint(0, height, size=(self.subsample_size, self.subsample_size)).astype(np.float32)
                elif self.subsample_type == "random2":
                    px = np.random.uniform(0, width - 1e-5, size=(self.subsample_size, self.subsample_size)).astype(
                        np.float32)
                    py = np.random.uniform(0, height - 1e-5, size=(self.subsample_size, self.subsample_size)).astype(
                        np.float32)
                else:
                    px, py = np.meshgrid(np.arange(width).astype(np.float32), np.arange(height).astype(np.float32))

                result["pixelcoords"] = np.stack((px, py), axis=-1)

        return result
