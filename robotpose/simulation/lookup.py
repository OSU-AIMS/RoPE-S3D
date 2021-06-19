# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import os
import random
import string
from typing import Union, Tuple

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from ..CompactJSONEncoder import CompactJSONEncoder
from ..paths import Paths as p
from ..projection import Intrinsics
from ..urdf import URDFReader
from ..utils import str_to_arr, get_key
from .render import Renderer

"""
ROBOT
"""
class RobotLookupCreator(Renderer):
    def __init__(self, camera_pose: np.ndarray, intrinsics: Union[str, Intrinsics]):
        self.inp_pose = camera_pose
        self.u_reader = URDFReader()
        super().__init__('seg', camera_pose=camera_pose, camera_intrin=intrinsics)

    def load_config(self, joints_to_render: int, angles_to_do: str, divisions:np.ndarray):
        self.num_rendered = joints_to_render
        self.setMaxParts(joints_to_render)
        self.divisions = np.array(divisions)
        self.angles_to_do = str_to_arr(angles_to_do)

        self.divisions[~self.angles_to_do] = 1
        self.num = int(np.prod(self.divisions))

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
            f.attrs['intrinsics'] = str(self.intrinsics)
            f.attrs['num_links_rendered'] = self.num_rendered
            f.attrs['angles_changed'] = self.angles_to_do
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


"""
Camera intrin
Camera pos
Joints used
    Joint max/min
    Div / joint
"""

class RobotLookupInfo():

    def __init__(self) -> None:
        self.update()
    
    def update(self):
        self.data = {}

        #paths = [os.path.join(r,x) for r,d,y in os.walk(p().ROBOT_LOOKUPS) for x in y if x.endswith('.h5')]
        paths = [os.path.join(p().ROBOT_LOOKUPS,x) for x in os.listdir(p().ROBOT_LOOKUPS) if x.endswith('.h5')]
        raw_tables = {}
        for path in paths:
            with h5py.File(path,'r') as f:
                a = dict(f.attrs)
            name = os.path.basename(os.path.normpath(path)).replace('.h5','')
            raw_tables[name] = a

        # Normalize all values
        for key in raw_tables:
            tmp_intrin = Intrinsics(raw_tables[key]['intrinsics'])
            raw_tables[key]['element_number'] = tmp_intrin.size * np.prod(raw_tables[key]['divisions'])
            raw_tables[key]['pose_number'] = np.prod(raw_tables[key]['divisions'])
            raw_tables[key]['intrinsics'] = str(tmp_intrin)
            raw_tables[key]['pose'] = tuple(raw_tables[key]['pose'])
            for attr in ['angles_changed', 'divisions']:
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

        for table in raw_tables:
            intrin = get_key(intrin_shortnames, raw_tables[table]['intrinsics'])
            pose = get_key(pose_shortnames, raw_tables[table]['pose'])
            self.data['lookups'][intrin][pose][table] = raw_tables[table]

        self._write()


    def _write(self):
        with open(p().ROBOT_LOOKUP_INFO,'w') as f:
            f.write(CompactJSONEncoder(max_width = 90, indent=4).encode(self.data).replace('\\','/'))



class RobotLookupManager(RobotLookupInfo):

    def __init__(self, element_bits = 32) -> None:
        self.element_bits = element_bits
        super().__init__()

    def get(self, 
        intrinsics: Union[str, Intrinsics], 
        camera_pose: np.ndarray, 
        num_rendered_links: int, 
        varying_angles: str, 
        max_elements: int = None,
        max_poses: int = None,
        divisions: np.ndarray = None,
        create_optimal: bool = True
        ):

        assert sum([x is not None for x in [max_elements, max_poses, divisions]]) > 0,\
             "Some specifying critera must be given in order to limit the size of the lookup requested"
        assert sum([x is not None for x in [max_elements, max_poses, divisions]]) == 1,\
             "Only one specifiying criterion can be used from [max_elements, max_poses, divisons]"

        varying_angles_arr = str_to_arr(varying_angles)

        if type(intrinsics) is str: intrinsics = Intrinsics(intrinsics)
        intrinsics = str(intrinsics)

        create = False

        if intrinsics in self.data['intrinsics'].values():
            intrinsic_short = get_key(self.data['intrinsics'], intrinsics)
            if tuple(list(camera_pose)) in self.data['camera_poses'].values():
                camera_pose_short = get_key(self.data['camera_poses'], tuple(list(camera_pose)))
            else:
                create = True
        else:
            create = True

        if not create:
            acceptable = self.data['lookups'][intrinsic_short][camera_pose_short]
            acceptable = {k:acceptable[k] for k in acceptable if acceptable[k]['num_links_rendered'] == num_rendered_links}
            acceptable = {k:acceptable[k] for k in acceptable if np.all([x != 1 for x in acceptable[k]['divisions']] == varying_angles_arr,-1)}
            if len(acceptable) == 0:
                create = True
            else:
                if max_elements is not None:
                    acceptable = {k:acceptable[k] for k in acceptable if acceptable[k]['element_number'] <= max_elements}
                elif max_poses is not None:
                    acceptable = {k:acceptable[k] for k in acceptable if acceptable[k]['pose_number'] <= max_poses}
                elif divisions is not None:
                    acceptable = {k:acceptable[k] for k in acceptable if acceptable[k]['divisions'] == list(divisions)}

            if len(acceptable) == 0:
                create = True
            else:
                if create_optimal:
                    pass

        if create:
            if divisions is None:
                if max_poses is None:
                    max_poses = max_elements / (Intrinsics(intrinsics).size * self.element_bits)
                # By default, allocate divisions equally
                divisions = np.zeros(6, int)
                divisions[varying_angles_arr] = int(max_poses ** (1 / sum(varying_angles_arr)))

            name = self.create(intrinsics, camera_pose, num_rendered_links, varying_angles, divisions)

        else:
            # Return one with highest pose count
            mx = max([x['pose_number'] for x in acceptable.values()])
            name = [k for k in acceptable if acceptable[k]['pose_number'] == mx][0]

        return self.load(name)


    def load(self, name: str):
        if not name.endswith('.h5'):
            name = name + '.h5'
        with h5py.File(os.path.join(p().ROBOT_LOOKUPS, name), 'r') as f:
            return np.copy(f['angles']), np.copy(f['depth'])


    def create(self, intrinsics: Union[str, Intrinsics], camera_pose: np.ndarray, num_rendered_links: int, varying_angles: str, divisions: np.ndarray):
        creator = RobotLookupCreator(camera_pose, intrinsics)
        creator.load_config(num_rendered_links, varying_angles, divisions)
        letters = string.ascii_lowercase
        pick = True
        while pick:
            name = ''.join(random.choice(letters) for i in range(5)) + ('.h5')
            if name not in os.listdir(p().ROBOT_LOOKUPS):
                pick = False
        creator.run(os.path.join(p().ROBOT_LOOKUPS, name), False)
        return name
