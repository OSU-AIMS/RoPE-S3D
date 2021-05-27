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
import os
import json
import string

import cv2

from ..projection import Intrinsics

from .render import Renderer
from ..urdf import URDFReader
from ..paths import Paths as p
from ..CompactJSONEncoder import CompactJSONEncoder

from tqdm import tqdm

class LookupCreator(Renderer):
    def __init__(self, camera_pose, ds_factor = 8):
        self.inp_pose = camera_pose
        self.ds_factor = ds_factor
        self.u_reader = URDFReader()
        super().__init__('seg', camera_pose=camera_pose, camera_intrin=f'1280_720_color_{ds_factor}')

    def load_config(self, joints_to_render, angles_to_do, divisions):
        self.setMaxParts(joints_to_render)
        self.divisions = np.array(divisions)
        self.angles_to_do = np.array(angles_to_do)

        self.divisions[~self.angles_to_do] = 1
        self.num = np.prod(self.divisions)

        self.angles = np.zeros((self.num,6))

        for idx in np.where(self.angles_to_do)[0]:
            angle_range = np.linspace(self.u_reader.joint_limits[idx,0],self.u_reader.joint_limits[idx,1],self.divisions[idx])

            repeat = np.prod(self.divisions[:idx])
            tile = self.num // (repeat * self.divisions[idx])

            self.angles[:,idx] = np.tile(np.repeat(angle_range,repeat),tile)

    def run(self, file_name, preview = True):

        self.setJointAngles([0,0,0,0,0,0])
        color, depth = self.render()

        depth_arr = np.zeros((self.num, *color.shape[:2]), dtype=float)

        for pose,idx in tqdm(zip(self.angles, range(len(self.angles))),total=len(self.angles),desc="Rendering Lookup Table"):
            self.setJointAngles(pose)
            color, depth = self.render()
            depth_arr[idx] = depth
            if preview: self._show(color)

        with tqdm(total=2, desc=f"Writing to {file_name}") as pbar:
            f = h5py.File(file_name, 'w')
            f.attrs['pose'] = self.inp_pose
            f.attrs['ds_factor'] = self.ds_factor
            f.attrs['intrinsics'] = str(self.intrinsics)
            f.attrs['joints_used'] = self.angles_to_do
            f.attrs['divisions'] = self.divisions
            f.create_dataset('angles', data=self.angles)
            pbar.update(1)
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



LOOKUP_INFO = os.path.join(p().LOOKUPS,'lookups.json')

"""
Camera intrin
Camera pos
Joints used
    Joint max/min
    Div / joint
"""

class LookupInfo():

    def __init__(self) -> None:
        self.update()
    
    def update(self):
        self.data = {}

        paths = [os.path.join(r,x) for r,d,y in os.walk(p().LOOKUPS) for x in y if x.endswith('.h5')]
        raw_tables = {}
        for path in paths:
            with h5py.File(path,'r') as f:
                a = dict(f.attrs)
            name = os.path.basename(os.path.normpath(path)).replace('.h5','')
            raw_tables[name] = a

        # Normalize all values
        for key in raw_tables:
            raw_tables[key]['intrinsics'] = str(Intrinsics(raw_tables[key]['intrinsics']))
            raw_tables[key]['pose'] = tuple(raw_tables[key]['pose'])
            for attr in ['joints_used', 'divisions']:
                raw_tables[key][attr] = list(raw_tables[key][attr])
            

        camera_poses = {x['pose'] for x in raw_tables.values()}
        pose_shortnames = {('P_'+k):v for k,v in zip(string.ascii_uppercase[:len(camera_poses)], camera_poses)}
        self.data['camera_poses'] = pose_shortnames

        intrins = {x['intrinsics'] for x in raw_tables.values()}
        intrin_shortnames = {('I_'+k):v for k,v in zip(string.ascii_uppercase[:len(intrins)], intrins)}
        self.data['intrinsics'] = intrin_shortnames

        # Create structure for lookup organization (intrin -> pose -> table)
        pose_dict = {pose:{} for pose in pose_shortnames}
        self.data['lookups'] = {intrin:pose_dict for intrin in intrin_shortnames}

        def get_key(dict, val):
            return list(dict.keys())[list(dict.values()).index(val)]

        for table in raw_tables:
            intrin = get_key(intrin_shortnames, raw_tables[table]['intrinsics'])
            pose = get_key(pose_shortnames, raw_tables[table]['pose'])
            self.data['lookups'][intrin][pose][table] = raw_tables[table]

        self._write()


    def _write(self):
        with open(LOOKUP_INFO,'w') as f:
            f.write(CompactJSONEncoder(max_width = 90, indent=4).encode(self.data).replace('\\','/'))

    def _read(self):
        if os.path.isfile(LOOKUP_INFO):
            with open(LOOKUP_INFO,'r') as f:
                return json.load(f)
        else:
            return {'camera_poses':{},'tables':{}}