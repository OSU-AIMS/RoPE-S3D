# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import os

import numpy as np
import pyrender
import trimesh

from ..urdf import URDFReader


def default_color_maker(num):
    b = np.linspace(0,255,num).astype(int) # Blue values are always unique

    g = [0] * b.size
    r = np.abs(255 - 2*b)

    colors = []
    for idx in range(num):
        colors.append([b[idx],g[idx],r[idx]])
    return colors

DEFAULT_COLORS = default_color_maker(7)


class MeshLoader():

    def __init__(self, include_t = False):

        self.ureader = URDFReader()
        if not include_t:
            self.name_list = self.ureader.mesh_names
            self.mesh_list = self.ureader.mesh_paths
        else:
            self.name_list = self.ureader.mesh_names[:-1]
            self.mesh_list = self.ureader.mesh_paths[:-1]

        self.load()

    def load(self):
        self._meshes = []
        for file in self.mesh_list:
            tm = trimesh.load(os.path.join(os.getcwd(),file))
            self._meshes.append(pyrender.Mesh.from_trimesh(tm,smooth=True))

    @property
    def meshes(self):
        return self._meshes

    @property
    def names(self):
        return self.name_list

    @property
    def meshes_and_names(self):
        return self._meshes, self.name_list


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


def makePose(x,y,z,pitch,roll,yaw):
    """Returns 4x4 pose array.
    Makes pose array given positon and angle
    """
    pose = angToPoseArr(yaw,pitch,roll)
    pose = translatePoseArr(x,y,z,pose)
    return pose


def setPoses(scene, nodes, poses):
    """
    Set all the poses of objects in a scene
    """
    for node, pose in zip(nodes,poses):
        scene.set_pose(node,pose)
