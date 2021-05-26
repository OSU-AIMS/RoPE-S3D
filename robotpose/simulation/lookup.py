# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import numpy as np
import h5py

import cv2

from .render import Renderer
from ..urdf import URDFReader
from ..paths import Paths as p

from tqdm import tqdm

class LookupCreator(Renderer):
    def __init__(self, camera_pose, ds_factor = 8):
        self.ds_factor = ds_factor
        self.u_reader = URDFReader()
        super().__init__('seg', camera_pose=camera_pose, camera_intrin=f'1280_720_color_{ds_factor}')

    def load_config(self, joints_to_render, angles_to_do, divisions):
        self.setMaxParts(joints_to_render)
        divisions = np.array(divisions)
        angles_to_do = np.array(angles_to_do)

        divisions[~angles_to_do] = 1
        self.num = np.prod(divisions)

        self.angles = np.zeros((self.num,6))

        for idx in np.where(angles_to_do)[0]:
            angle_range = np.linspace(self.u_reader.joint_limits[idx,0],self.u_reader.joint_limits[idx,1],divisions[idx])

            repeat = np.prod(divisions[:idx])
            tile = self.num//(repeat*divisions[idx])

            self.angles[:,idx] = np.tile(np.repeat(angle_range,repeat),tile)

    def run(self, file_name, preview = True):

        self.setJointAngles([0,0,0,0,0,0])
        color, depth = self.render()

        #color_arr = np.zeros((self.num, *color.shape), dtype=np.uint8)
        depth_arr = np.zeros((self.num, *color.shape[:2]), dtype=float)

        for pose,idx in tqdm(zip(self.angles, range(len(self.angles))),total=len(self.angles),desc="Rendering Lookup Table"):
            self.setJointAngles(pose)
            color, depth = self.render()
            #color_arr[idx] = color
            depth_arr[idx] = depth
            if preview: self._show(color)

        with tqdm(total=2, desc=f"Writing to {file_name}") as pbar:
            f = h5py.File(file_name, 'w')
            f.attrs['pose'] = self.camera_pose
            f.attrs['ds_factor'] = self.ds_factor
            f.attrs['intrinsics'] = str(self.intrinsics)
            f.create_dataset('angles', data=self.angles)
            pbar.update(1)
            # f.create_dataset('color',data=color_arr, compression="gzip", compression_opts=1)
            # pbar.update(1)
            f.create_dataset('depth', data=depth_arr, compression="gzip", compression_opts=1)
            pbar.update(1)


    def _show(self, color):
        size = color.shape[0:2]
        dim = [x*8 for x in size]
        dim.reverse()
        dim = tuple(dim)
        color = cv2.resize(color, dim, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Lookup Table Creation",color)
        cv2.waitKey(1)



class LookupInfo():
    def __init__(self) -> None:
        self._update()
    
    def _update(self):
        pass