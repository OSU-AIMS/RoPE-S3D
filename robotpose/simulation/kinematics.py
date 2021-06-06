# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley
# Based on "FwdKinematic_MH5L_AllJoints" authored by acbuynak

import json
import os
import time

import numpy as np

from ..paths import Paths as p
from ..urdf import URDFReader

from klampt import WorldModel

# class ForwardKinematics():

#     def __init__(self) -> None:
#         self.load()

#     def load(self):
#         while not self._getParams(): pass

#         self.aa    = np.array(self.params['a'])
#         self.alpha = np.array(self.params['alpha'])
#         self.dd    = np.array(self.params['d'])

#     def _getParams(self):
#         u_reader = URDFReader()

#         if os.path.isfile(p().DH_PARAMS):
#             with open(p().DH_PARAMS,'r') as f:
#                 config = json.load(f)
#         else:
#             config = {}

#         if u_reader.name in config:
#             self.params = config[u_reader.name]
#             return True
#         else:
#             u_reader.guessDHParams()
#             return False

#     def calc(self, p_in):
#         """
#         Performs Forward Kinematic Calculation to find the xyz (euler) position of each joint. Rotations NOT output.
#         Based on FwdKinematic_MH5L_AllJoints created by acbuynak.
#         Method: Denavit-Hartenberg parameters used to generate Transformation Matrices. Translation points extracted from the TF matrix.
#         :param p_in: List of 6 joint angles (radians)
#         :return vectors: List Numpy Array (6x3) where each row is xyz origin of joints
#         """
#         def bigMatrix(a, alpha, d, pheta):
#             cosp = np.cos(pheta)
#             sinp = np.sin(pheta)
#             cosa = np.cos(alpha)
#             sina = np.sin(alpha)
#             T = np.array([[cosp, -sinp, 0, a],
#                             [sinp * cosa, cosp * cosa, -sina, -d * sina],
#                             [sinp * sina, cosp * sina, cosa, d * cosa],
#                             [0, 0, 0, 1]])
#             return T

#         pheta = [0, p_in[0], p_in[1]-1.57079, -p_in[2], p_in[3], p_in[4], p_in[5]]

#         ## Forward Kinematics
#         # Ref. L13.P5
#         T_x = np.zeros((6,4,4))
#         T_x[0] = bigMatrix(self.aa[0], self.alpha[0], self.dd[1], pheta[1])

#         for i in range(1,6):    # Apply transforms in order, yielding each sucessive frame
#             T_x[i] = np.matmul(T_x[i-1], bigMatrix(self.aa[i], self.alpha[i], self.dd[i+1], pheta[i+1]))

#         # Extract list of vectors between frames
#         vectors = T_x[:, :-1, 3]

#         return vectors



def angToPoseArr(yaw,pitch,roll, arr = None):
    """Returns 4x4 pose array.
    Converts rotations to a pose array
    """
    # Takes pitch, roll, yaw and converts into a pose arr
    angs = np.array([yaw,pitch,roll])
    c = np.cos(angs)
    s = np.sin(angs)
    if arr is None:
        pose = np.zeros((4,4))
    else:
        pose = arr

    pose[0,0] = c[0] * c[1]
    pose[1,0] = c[1] * s[0]
    pose[2,0] = -1 * s[1]

    pose[0,1] = c[0] * s[1] * s[2] - c[2] * s[0]
    pose[1,1] = c[0] * c[2] + np.prod(s)
    pose[2,1] = c[1] * s[2]

    pose[0,2] = s[0] * s[2] + c[0] * c[2] * s[1]
    pose[1,2] = c[2] * s[0] * s[1] - c[0] * s[2]
    pose[2,2] = c[1] * c[2]

    pose[3,3] = 1.0

    return pose
    

def translatePoseArr(x,y,z, arr = None):
    """Returns 4x4 pose array.
    Translates a pose array
    """
    if arr is None:
        pose = np.zeros((4,4))
    else:
        pose = arr

    pose[0,3] = x
    pose[1,3] = y
    pose[2,3] = z

    return pose


#def makePose(x,y,z,pitch,roll,yaw):
def makePose(x,y,z,roll,pitch,yaw):
    """Returns 4x4 pose array.
    Makes pose array given positon and angle
    """
    pose = angToPoseArr(yaw,pitch,roll)
    pose = translatePoseArr(x,y,z,pose)
    return pose



class ForwardKinematics():

    def __init__(self) -> None:
        self.load()

    def load(self):
        u_reader = URDFReader()
        self.world = WorldModel()

        # Load into env
        self.world.loadElement(u_reader.path)
        self.robot = self.world.robot(0)

        # Get link IDs
        link_ids = [self.robot.link(idx).getName() for idx in range(self.robot.numLinks())]
        # Get mapping
        self.link_map = {k:link_ids.index(k) for k in u_reader.mesh_names}
        self.link_idxs = [x for x in self.link_map.values()]


    def calc(self, p_in):
        h = time.time()
        angs = np.zeros(self.robot.numLinks())
        angs[self.link_idxs[1:]] = p_in # base link does not have angle
        
        self.robot.setConfig(angs)
        

        poses = np.zeros((7,4,4))
        
        for idx,i in zip(self.link_idxs, range(len(self.link_idxs))):
            trans = self.robot.link(idx).getTransform()
            poses[i,3,3] = 1
            poses[i,:3,3] = trans[1]
            poses[i,:3,:3] = np.reshape(trans[0],(3,3),'F')

        print(time.time()-h)

        return poses
