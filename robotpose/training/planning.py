# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley


from typing import List, Union

import numpy as np

from ..urdf import URDFReader
from ..utils import str_to_arr


class Planner():

    def __init__(self):
        self.u_reader = URDFReader()

    def basicGrid(self, varying_joints: str, max_poses: int):

        varying_joints_arr = str_to_arr(varying_joints)
        # By default, allocate divisions equally
        divisions = np.zeros(6, int)
        divisions[varying_joints_arr] = int(max_poses ** (1 / sum(varying_joints_arr)))

        pose_divs = []
        for idx in range(6):
            if divisions[idx] == 0:
                pose_divs.append([0])
            else:
                pose_divs.append(np.linspace(*self.u_reader.joint_limits[idx],num=divisions[idx]).tolist())

        return self._uniformGrid(pose_divs)


    def noisyGrid(self, varying_joints: str, max_poses: int, noise: Union[float, list, np.ndarray]):
        base = self.basicGrid(varying_joints, max_poses)

        if type(noise) is float:
            noise = [noise for i in range(6)]
        noise = np.array(noise)

        varying_joints_arr = str_to_arr(varying_joints)
        noise *= varying_joints_arr

        noise_arr = np.random.uniform(-noise,noise,(base.shape[0],6))

        out = base + noise_arr

        self.poses = np.clip(out,self.u_reader[:,0],self.u_reader[:,1])
        return self.poses


    def _uniformGrid(self, joint_poses: List[List[float]]):

        num = np.prod([len(x) for x in joint_poses])

        types = [[joint_poses[i],joint_poses[i][::-1]] for i in range(1,6)]

        states = np.array([False for i in range(5)],bool)

        def get_divs(joint_idx):
            # Use joint index convention
            return types[joint_idx-1][int(states[joint_idx-1])]

        def change_state(joint_idx):
            # Use joint index convention
            states[joint_idx-1] = ~states[joint_idx-1]
            
        poses = np.zeros((num,6),float)

        idx = 0

        for s_pos in joint_poses[0]:
            for l_pos in get_divs(1):
                for u_pos in get_divs(2):
                    for r_pos in get_divs(3):
                        for b_pos in get_divs(4):
                            for t_pos in get_divs(5):
                                poses[idx] = [s_pos,l_pos,u_pos,r_pos,b_pos,t_pos]
                                idx += 1
                            change_state(5)
                        change_state(4)
                    change_state(3)
                change_state(2)
            change_state(1)
        

        self.poses = poses
        return poses
