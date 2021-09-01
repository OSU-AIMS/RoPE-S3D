# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from typing import Union
import numpy as np
from klampt import WorldModel

from ..urdf import URDFReader


class ForwardKinematics():
    """Calculates the forward kinematics of the robot in the active URDF"""

    def __init__(self) -> None:
        self.load()

    def load(self):
        u_reader = URDFReader()
        self.world = WorldModel(u_reader.path)
        self.robot = self.world.robot(0)

        # Get link IDs
        link_ids = [self.robot.link(idx).getName() for idx in range(self.robot.numLinks())]

        # Get mapping
        self.link_map = {k:link_ids.index(k) for k in u_reader.mesh_names}
        self.link_idxs = [x for x in self.link_map.values()]


    def calc(self, p_in: Union[list, np.ndarray]):
        """Calculate mesh poses based on joint angles"""

        angs = np.zeros(self.robot.numLinks())
        angs[self.link_idxs[1:]] = p_in # base link does not have angle
        
        # Set angles
        self.robot.setConfig(angs)
    
        poses = np.zeros((7,4,4))
        
        # Get pose
        for idx,i in zip(self.link_idxs, range(len(self.link_idxs))):
            trans = self.robot.link(idx).getTransform()
            poses[i,3,3] = 1
            poses[i,:3,3] = trans[1]
            poses[i,:3,:3] = np.reshape(trans[0],(3,3),'F') # Use Fortran mapping for reshape

        # Returns N x 4 x 4 array
        return poses
