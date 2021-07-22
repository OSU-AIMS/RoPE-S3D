# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from typing import Union

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from .constants import (CROP_MAX_PER_JOINT, CROP_PADDING,
                        CROP_RENDER_WEIGHTING, CROP_SEC_ALLOTTED_APPROX,
                        CROP_VARYING, MAX_LINKS)
from .paths import Paths
from .projection import Intrinsics
from .simulation.render import Renderer
from .urdf import URDFReader
from .utils import get_extremes, str_to_arr

# Robot > Pose > Intrinsics 

class Crop(Renderer):
    def __init__(self, camera_pose: np.ndarray, intrinsics: Union[str, Intrinsics]):
        self.u_reader = URDFReader()
        create = False
        with h5py.File(Paths().CROP_DATA,'a') as f:
            name = f'{self.u_reader.name}/{self._list_to_str(camera_pose)}/{intrinsics}'
            if name not in f:
                f.create_dataset(name,shape=(MAX_LINKS,4),dtype=int)
                create = True
            else:
                if not np.any(f[name]):
                    create = True

        if create:
            super().__init__('seg', camera_pose, intrinsics)
            self._create(name)

        self.load(name)

    def _create(self, name):
        data = np.zeros((MAX_LINKS,4),int)

        self.setJointAngles([0,0,0,0,0,0])
        self.setMaxParts(1)
        color, depth = self.render()
        data[1] = self._calculate_crop(depth)

        for self.num_links in range(2,MAX_LINKS):
            self.setMaxParts(self.num_links)
            self.setJointAngles([0,0,0,0,0,0])
            self._generate_angles()
            depth_arr = np.zeros(color.shape[:2], dtype=float)

            for pose,idx in tqdm(zip(self.angles, range(len(self.angles))),total=len(self.angles),desc="Finding Crop"):
                self.setJointAngles(pose)
                color, depth = self.render()
                depth_arr += depth * .01
                if idx % 100 == 0:
                    self._show(depth_arr)
                
            data[self.num_links] = self._calculate_crop(depth_arr, process_sum=False)

        data[0] = data[-1]

        print(data)
        with h5py.File(Paths().CROP_DATA,'a') as f:
            f[name][:] = data


    def _list_to_str(self, lst):
        out = "["
        for item in lst:
            out += f" {item:.4f}"
        return out + " ]"


    def _calculate_crop(self, depth: np.ndarray, process_sum = False):
        d = np.sum(depth,0) if process_sum else depth
        d = d != 0
        extremes = get_extremes(d)
        out = np.zeros((4,))
        out[0] = max(np.min(extremes[0]) - CROP_PADDING ,0) # Min r
        out[1] = min(np.max(extremes[1]) + CROP_PADDING, self.intrinsics.height - 1)    # Max r
        out[2] = max(np.min(extremes[2]) - CROP_PADDING, 0) # Min c
        out[3] = min(np.max(extremes[3]) + CROP_PADDING, self.intrinsics.width - 1) # Max c
        return out


    def _generate_angles(self):

        div_weighting = np.array((CROP_RENDER_WEIGHTING)[:self.num_links - 1])
        div_weighting = div_weighting / np.sum(div_weighting)

        # num_poses = CROP_MAX_POSE_MULT * np.sqrt(self.intrinsics.size)
        num_poses = CROP_SEC_ALLOTTED_APPROX / (self.intrinsics.size * 1.2*(10**-8) + .002)

        base_div = div_weighting * ((num_poses / np.prod(div_weighting[div_weighting != 0])) ** (1 / len(div_weighting[div_weighting != 0])))
        base_div[base_div < 1] = 1
        base_div[base_div > CROP_MAX_PER_JOINT] = CROP_MAX_PER_JOINT
        base_div = base_div.astype(int)

        self.divisions = np.ones((6,),dtype=int)
        self.divisions[:self.num_links - 1] = base_div
        print(self.divisions)
        self.num = np.prod(self.divisions)

        self.angles = np.zeros((self.num,6))

        for idx in np.where(str_to_arr(CROP_VARYING))[0]:
            angle_range = np.linspace(self.u_reader.joint_limits[idx,0],self.u_reader.joint_limits[idx,1],self.divisions[idx])

            repeat = np.prod(self.divisions[:idx])
            tile = self.num // (repeat * self.divisions[idx])

            self.angles[:,idx] = np.tile(np.repeat(angle_range,repeat),tile)


    def _show(self, color: np.ndarray):
        cv2.imshow("Crop Calculation",color)
        cv2.waitKey(1)
        
    def load(self, name: str):
        with h5py.File(Paths().CROP_DATA,'r') as f:
            self.data = np.copy(f[name])

    def __getitem__(self, key):
        if key is None:
            key = 0
        elif type(key) is not int:
            key = int(key)

        return self.data[key]

    def size(self, key):
        crop = self.data[key]
        return (crop[1] - crop[0]) * (crop[3] - crop[2])


def applyCrop(mat, crop):
    return mat[crop[0]:crop[1]+1,crop[2]:crop[3]+1]

def applyBatchCrop(mat, crop):
    return mat[:,crop[0]:crop[1]+1,crop[2]:crop[3]+1]
