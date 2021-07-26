# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from typing import List, Union

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


class Crop(Renderer):
    """Specifies theoretical image bounds of robot in image given camera data"""

    def __init__(self, camera_pose: np.ndarray, intrinsics: Union[str, Intrinsics]):
        self.u_reader = URDFReader()
        create = False
        with h5py.File(Paths().CROP_DATA,'a') as f:
            # Name is just data as a string
            # Robot > Pose > Intrinsics 
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

    def _create(self, name: str):
        """Find crop data and store to .h5 file"""
        data = np.zeros((MAX_LINKS,4),int)  # Crop data

        # Calculate crop for just base
        self.setJointAngles([0,0,0,0,0,0])
        self.setMaxParts(1)
        color, depth = self.render()
        data[1] = self._calculate_crop(depth)

        # Calculate crop for all links other than base
        for self.num_links in range(2,MAX_LINKS):

            # Reset robot for current number of links
            self.setMaxParts(self.num_links)
            self.setJointAngles([0,0,0,0,0,0])

            self._generate_angles() # Figure out the poses to render
            depth_arr = np.zeros(color.shape[:2], dtype=float)  # Init depth sum arr

            for pose,idx in tqdm(zip(self.angles, range(len(self.angles))),total=len(self.angles),desc=f"Finding Crop {self.divisons}"):
                # Render; add to sum arr
                self.setJointAngles(pose)
                color, depth = self.render()
                depth_arr += depth * .01    # For visual, weight each depth less so it looks nicer

                # Show a preview every 100 renderings
                if idx % 100 == 0:
                    self._show(depth_arr)
            cv2.destroyAllWindows()
                
            data[self.num_links] = self._calculate_crop(depth_arr, process_sum=False)

        # 0 index is the max area of crops
        data[0] = data[-1]

        with h5py.File(Paths().CROP_DATA,'a') as f:
            f[name][:] = data


    def _list_to_str(self, lst: list) -> str:
        """Uniformly format a list into a string"""
        out = "["
        for item in lst:
            out += f" {item:.4f}"
        return out + " ]"


    def _calculate_crop(self, depth: np.ndarray, process_sum: bool = False) -> List[int]:
        """Calculate a crop from a sum of depths"""
        d = np.sum(depth,0) if process_sum else depth

        # Find areas where there is any depth data
        d = d != 0
        extremes = get_extremes(d)  # Get the edges of this data
        out = np.zeros((4,))

        # Add padding and make sure not outside iamge
        out[0] = max(np.min(extremes[0]) - CROP_PADDING ,0) # Min r
        out[1] = min(np.max(extremes[1]) + CROP_PADDING, self.intrinsics.height - 1)    # Max r
        out[2] = max(np.min(extremes[2]) - CROP_PADDING, 0) # Min c
        out[3] = min(np.max(extremes[3]) + CROP_PADDING, self.intrinsics.width - 1) # Max c
        return out


    def _generate_angles(self):
        """Create an array of angles to render"""
        # Get the importance of each joint in rendering
        div_weighting = np.array((CROP_RENDER_WEIGHTING)[:self.num_links - 1])
        div_weighting = div_weighting / np.sum(div_weighting)

        # Figure out how many poses can be rendered in the allowed time
        # Configured for GTX 1070 and Ryzen 2600
        num_poses = CROP_SEC_ALLOTTED_APPROX / (self.intrinsics.size * 1.2*(10**-8) + .002)

        # Calculate the number of divisons rendered for each joint
        base_div = div_weighting * ((num_poses / np.prod(div_weighting[div_weighting != 0])) ** (1 / len(div_weighting[div_weighting != 0])))
        base_div[base_div < 1] = 1  # Smallest allowed is 1 division
        base_div[base_div > CROP_MAX_PER_JOINT] = CROP_MAX_PER_JOINT    # Limit how many poses a single joint can be in
        base_div = base_div.astype(int) # Round down for all

        # If not being rendered, set to 1 div
        self.divisions = np.ones((6,),dtype=int)
        self.divisions[:self.num_links - 1] = base_div

        self.num = np.prod(self.divisions)  # How many poses this is going to create

        self.angles = np.zeros((self.num,6))

        for idx in np.where(str_to_arr(CROP_VARYING))[0]:   # Ignore joints that are overridden for crops
            # Divison locations
            angle_range = np.linspace(self.u_reader.joint_limits[idx,0],self.u_reader.joint_limits[idx,1],self.divisions[idx])

            repeat = np.prod(self.divisions[:idx])  # How may times to repeat instances
            tile = self.num // (repeat * self.divisions[idx])   # How many times to tile after repeating

            self.angles[:,idx] = np.tile(np.repeat(angle_range,repeat),tile)    # Set


    def _show(self, color: np.ndarray):
        """Show a preview of the rendering"""
        cv2.imshow("Crop Calculation",color)
        cv2.waitKey(1)
        
    def load(self, name: str):
        """Get already-compiled crop data from the .h5 file"""
        with h5py.File(Paths().CROP_DATA,'r') as f:
            self.data = np.copy(f[name])

    def __getitem__(self, key: int) -> List[int]:
        """Return a crop for a number of joints visible"""
        if key is None:
            key = 0
        elif type(key) is not int:
            key = int(key)

        return self.data[key]

    def size(self, n: int) -> int:
        """Get the size of the crop (in px) for n joints visible"""
        crop = self.data[n]
        return (crop[1] - crop[0]) * (crop[3] - crop[2])


def applyCrop(mat: np.ndarray, crop: List[int]) -> np.ndarray:
    """Apply a crop to a single image"""
    return mat[crop[0]:crop[1]+1,crop[2]:crop[3]+1]

def applyBatchCrop(mat: np.ndarray, crop: List[int]) -> np.ndarray:
    """Apply a crop to an array of images"""
    return mat[:,crop[0]:crop[1]+1,crop[2]:crop[3]+1]
