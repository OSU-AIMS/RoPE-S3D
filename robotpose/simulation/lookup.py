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
from typing import List, Union

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from ..CompactJSONEncoder import CompactJSONEncoder
from ..constants import LOOKUP_NAME_LENGTH, GPU_MEMORY_ALLOWED_FOR_LOOKUP
from ..crop import Crop, applyBatchCrop
from ..paths import Paths as p
from ..projection import Intrinsics
from ..urdf import URDFReader
from ..utils import get_gpu_memory, get_key, str_to_arr
from .render import Renderer


class RobotLookupCreator(Renderer):
    """Creates a file that stores rendered depth data in a form that can be processed easily"""

    def __init__(self, camera_pose: np.ndarray, intrinsics: Union[str, Intrinsics]):
        self.inp_pose = camera_pose
        self.u_reader = URDFReader()
        self.croppper = Crop(camera_pose, intrinsics)
        super().__init__('seg', camera_pose=camera_pose, camera_intrin=intrinsics)

    def load_config(self, joints_to_render: int, angles_to_do: Union[str,np.ndarray], divisions:np.ndarray):
        """Load specifications for the lookup table"""

        self.num_rendered = joints_to_render
        self.setMaxParts(joints_to_render)
        self.crop = self.croppper[joints_to_render]
        
        # Convert to ndarray if needed
        self.angles_to_do = str_to_arr(angles_to_do) if type(angles_to_do) is str else angles_to_do

        # Load in divisions
        self.divisions = np.array(divisions)
        self.divisions[~self.angles_to_do] = 1
        self.num = int(np.prod(self.divisions))

        self._generate_angles() # Create angle set

    def _generate_angles(self):
        """Generate a set of angles based on divisons"""
        self.angles = np.zeros((self.num,6))

        for idx in np.where(self.angles_to_do)[0]:
            angle_range = np.linspace(self.u_reader.joint_limits[idx,0],self.u_reader.joint_limits[idx,1],self.divisions[idx])

            repeat = np.prod(self.divisions[:idx])
            tile = self.num // (repeat * self.divisions[idx])

            self.angles[:,idx] = np.tile(np.repeat(angle_range,repeat),tile)
    

    def _generate_depth_array(self, preview: bool = True) -> np.ndarray:
        """Render the robot at each of the specified poses"""
        
        # Get size of images
        self.setJointAngles([0,0,0,0,0,0])
        color, depth = self.render()
        depth_arr = np.zeros((self.num, *color.shape[:2]), dtype=float)

        # Render the lookup
        for pose,idx in tqdm(zip(self.angles, range(len(self.angles))),total=len(self.angles),desc=f"Rendering {list(self.divisions)} Lookup"):
            self.setJointAngles(pose)   # Set pose
            color, depth = self.render()    #Render
            depth_arr[idx] = depth  # Add to array
            if preview: 
                self._show(color)

        return depth_arr


    def run(self, file_name: str, preview: bool = True):
        """Create a new lookup"""

        depth_arr = self._generate_depth_array(preview) # Create lookup data
        depth_arr = applyBatchCrop(depth_arr, self.crop)    # Crop

        # Save to a new .h5 file    
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


    def _show(self, color: np.ndarray):
        size = color.shape[0:2]
        dim = [x*1 for x in size]
        dim.reverse()
        dim = tuple(dim)
        color = cv2.resize(color, dim, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Lookup Table Creation",color)
        cv2.waitKey(1)


class RobotLookupInfo():
    """Manages information about the currently-available lookups"""

    def __init__(self) -> None:
        self.update()

    def update(self):
        self.data = {}  # Clear data

        # Find all .h5 files except the crop data file
        paths = [os.path.join(p().ROBOT_LOOKUPS,x) for x in os.listdir(p().ROBOT_LOOKUPS) if x.endswith('.h5') and not os.path.basename(p().CROP_DATA) in x]
        raw_tables = {}
        
        # Create dict of attributes
        for path in paths:
            with h5py.File(path,'r') as f:
                a = dict(f.attrs)
            name = os.path.basename(os.path.normpath(path)).replace('.h5','')
            raw_tables[name] = a

        # Normalize all values' format
        for key in raw_tables:
            tmp_intrin = Intrinsics(raw_tables[key]['intrinsics'])
            raw_tables[key]['element_number'] = tmp_intrin.size * np.prod(raw_tables[key]['divisions'])
            raw_tables[key]['pose_number'] = np.prod(raw_tables[key]['divisions'])
            raw_tables[key]['intrinsics'] = str(tmp_intrin)
            raw_tables[key]['pose'] = tuple(raw_tables[key]['pose'])
            for attr in ['angles_changed', 'divisions']:
                raw_tables[key][attr] = list(raw_tables[key][attr])

        # Create abbreviation for poses
        camera_poses = {x['pose'] for x in raw_tables.values()}
        pose_shortnames = {('P_'+k):v for k,v in zip(string.ascii_uppercase[:len(camera_poses)], camera_poses)}
        self.data['camera_poses'] = pose_shortnames

        # Create abbreviation for intrinsics
        intrins = {x['intrinsics'] for x in raw_tables.values()}
        intrin_shortnames = {('I_'+k):v for k,v in zip(string.ascii_uppercase[:len(intrins)], intrins)}
        self.data['intrinsics'] = intrin_shortnames

        # Create structure for lookup organization (intrin -> pose -> table)
        self.data['lookups'] = {intrin:{pose:dict() for pose in pose_shortnames} for intrin in intrin_shortnames}

        for table in raw_tables:
            intrin = get_key(intrin_shortnames, raw_tables[table]['intrinsics'])
            pose = get_key(pose_shortnames, raw_tables[table]['pose'])
            self.data['lookups'][intrin][pose][table] = raw_tables[table]

        self._write()


    def _write(self):
        with open(p().ROBOT_LOOKUP_INFO,'w') as f:
            f.write(CompactJSONEncoder(max_width = 90, indent=4).encode(self.data).replace('\\','/'))



class RobotLookupManager(RobotLookupInfo):
    """Used to load and create lookup tables"""

    def __init__(self, element_bits: int = 32) -> None:
        self.element_bits = element_bits
        super().__init__()

    def get(self,
        intrinsics: Union[str, Intrinsics],
        camera_pose: np.ndarray,
        num_rendered_links: int,
        varying_angles: Union[str,np.ndarray],
        max_elements: int = None,
        max_poses: int = None,
        divisions: np.ndarray = None
        ) -> List[np.ndarray]:
        """Get a lookup table, creating one if needed

        Parameters
        ----------
        intrinsics : Union[str, Intrinsics], camera_pose : np.ndarray
            Camera descriptors
        num_rendered_links : int
            Number of links to render
        varying_angles : Union[str,np.ndarray]
            The angles that vary in this lookup

        At most one of the following
        ----------------------------
        max_elements : int, (calculated by default)
            Max number of array elements
        max_poses : int, optional
            Max number of robot poses
        divisions : np.ndarray, optional
            Exact angle space divisions

        Returns
        -------
        angles, depths
            ndarrays of information
        """

        self.update()

        # Make sure criteria are present, otherwise calc max elements
        assert sum([x is not None for x in [max_elements, max_poses, divisions]]) <= 1,\
             "Only one specifiying criterion can be used from [max_elements, max_poses, divisons]"
        if sum([x is not None for x in [max_elements, max_poses, divisions]]) == 0:
            max_elements = int(get_gpu_memory()[0] * GPU_MEMORY_ALLOWED_FOR_LOOKUP)

        # Convert varying angs to array if needed
        varying_angles_arr = str_to_arr(varying_angles) if type(varying_angles) is str else varying_angles

        # Make sure intrinsics string is correctly formatted
        intrinsics = str(Intrinsics(intrinsics))

        # See if there is any possibility of having a correct lookup
        create = False
        if intrinsics in self.data['intrinsics'].values():
            intrinsic_short = get_key(self.data['intrinsics'], intrinsics)
            if tuple(list(camera_pose)) in self.data['camera_poses'].values():
                camera_pose_short = get_key(self.data['camera_poses'], tuple(list(camera_pose)))
            else:
                create = True
        else:
            create = True

        # If there could possibly be a correct lookup, try to find it
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

        
        if create:
            # If no optimal lookup is present, make one
            if divisions is None:
                c = Crop(camera_pose, intrinsics)
                if max_poses is None:
                    max_poses = max_elements / (c.size(num_rendered_links) * self.element_bits)

                    print(max_poses*c.size(num_rendered_links)*self.element_bits)
                # By default, allocate divisions equally
                divisions = np.zeros(6, int)
                divisions[varying_angles_arr] = int(max_poses ** (1 / sum(varying_angles_arr)))

            name = self.create(intrinsics, camera_pose, num_rendered_links, varying_angles, divisions)
            self.update()
        else:
            # Return lookup with highest pose count
            mx = max([x['pose_number'] for x in acceptable.values()])
            name = [k for k in acceptable if acceptable[k]['pose_number'] == mx][0]

        return self.load(name)


    def load(self, name: str) -> List[np.ndarray]:
        """Return values from lookup table based on name"""
        if not name.endswith('.h5'):
            name = name + '.h5'
        with h5py.File(os.path.join(p().ROBOT_LOOKUPS, name), 'r') as f:
            return np.copy(f['angles']), np.copy(f['depth'])


    def create(self, 
        intrinsics: Union[str, Intrinsics], 
        camera_pose: np.ndarray, 
        num_rendered_links: int, 
        varying_angles: str, 
        divisions: np.ndarray
        ) -> str:
        """Create a new lookup"""
        
        creator = RobotLookupCreator(camera_pose, intrinsics)
        creator.load_config(num_rendered_links, varying_angles, divisions)

        # Create a unique name
        letters = string.ascii_lowercase
        pick = True
        while pick:
            name = ''.join(random.choice(letters) for i in range(LOOKUP_NAME_LENGTH)) + ('.h5')
            if name not in os.listdir(p().ROBOT_LOOKUPS):
                pick = False
        
        # Create lookup table
        creator.run(os.path.join(p().ROBOT_LOOKUPS, name), False)
        return name
