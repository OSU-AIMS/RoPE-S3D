# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import numpy as np
import os
import json

import pyrender
import trimesh

from ..paths import Paths as p
from ..urdf import URDFReader
from ..CompactJSONEncoder import CompactJSONEncoder


MESH_CONFIG = p().MESH_CONFIG

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

    def __init__(self):

        self.ureader = URDFReader()

        if not os.path.isfile(MESH_CONFIG):
            info = {}
            with open(MESH_CONFIG,'w') as f:
                json.dump(info, f, indent=4)

        self.refresh()
        self.load()

    def refresh(self):
        with open(MESH_CONFIG,'r') as f:
            d = json.load(f)

        if self.ureader.name not in d:
            d[self.ureader.name] = self.ureader.guessPoseConfig()
            with open(MESH_CONFIG,'w') as f:
                f.write(CompactJSONEncoder(max_width=90,precise=True,indent=4).encode(d))

        self.name_list = self.ureader.mesh_names[:-1]
        self.mesh_list = self.ureader.meshes[:-1]
        self.pose_list = [d[self.ureader.name][x] for x in self.name_list]

    def load(self):
        self._meshes = []
        for file, pose in zip(self.mesh_list, self.pose_list):
            tm = trimesh.load(os.path.join(os.getcwd(),file))
            self._meshes.append(pyrender.Mesh.from_trimesh(tm,smooth=True, poses=makePose(*pose)))

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


def posesFromData(ang, pos):
    """
    Returns Zx6x4x4 array of pose arrays
    """
    #Make arr in x,y,z,roll,pitch,yaw format
    coord = np.zeros((pos.shape[0],6,6))
    # Yaw of S-R is just S angle
    for idx in range(1,5):
        coord[:,idx,5] = ang[:,0]

    coord[:,1:6,0:3] = pos[:,:5]    # Set xyz translations

    coord[:,2,3] = ang[:,1]             # Pitch of L
    coord[:,3,3] = ang[:,1] - ang[:,2]  # Pitch of U
    coord[:,4,3] = ang[:,1] - ang[:,2]  # Pitch of R       

    coord[:,4,4] = -ang[:,3]    # Roll of R


    poses = np.zeros((coord.shape[0],6,4,4))    # X frames, 6 joints, 4x4 pose for each
    for idx in range(coord.shape[0]):
        for sub_idx in range(5):
            poses[idx,sub_idx] = makePose(*coord[idx,sub_idx])

    # Determine BT with vectors because it's easier
    bt = pos[:,5] - pos[:,4]    # Vectors from B to T

    y = poses[:,4,:3,1] # Y Axis is common with R joint
    z = np.cross(bt, y)

    z = z / np.vstack([np.linalg.norm(z, axis=-1)]*3).transpose()
    x = bt / np.vstack([np.linalg.norm(bt, axis=-1)]*3).transpose()

    b_poses = np.zeros((pos.shape[0],4,4))

    b_poses[:,:3,0] = x # X Unit
    b_poses[:,:3,1] = y # Y Unit
    b_poses[:,:3,2] = z # Z Unit
    b_poses[:,3,3] = 1
    b_poses[:,:3,3] = pos[:,4] # XYZ Offset

    poses[:,-1,:] = b_poses

    return poses


def setPoses(scene, nodes, poses):
    """
    Set all the poses of objects in a scene
    """
    for node, pose in zip(nodes,poses):
        scene.set_pose(node,pose)