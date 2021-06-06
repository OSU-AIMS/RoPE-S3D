# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .paths import Paths

def str_to_list(string):
    return [float(x) for x in string.split(' ')]


class URDFReader():
    def __init__(self):
        if self._get_path():
            self.load()

    def _get_path(self):
        p = Paths()
        if hasattr(p,'URDF'):
            self.internal_path = p.URDF
            return True
        else:
            return False

    def load(self):
        tree = ET.parse(self.internal_path)
        root = tree.getroot()

        # Find mesh locations and names
        self.meshes = []
        for link in root.findall('link')[:7]:
            self.meshes.append(link.find('visual').find('geometry').find('mesh').get('filename'))
        if sys.platform == 'win32':
            fileend = 'stl'
        elif sys.platform == 'linux':
            fileend = 'STL'
        self.meshes = [os.path.join('urdf',x.replace('package://',"").replace('STL',fileend)) for x in self.meshes]
        self.mesh_names = [os.path.splitext(os.path.basename(x))[0] for x in self.meshes]

        # Find joint limits
        self.joint_limits = []
        for joint in root.findall('joint')[:6]:
            jo = joint.find('limit')
            self.joint_limits.append([float(jo.get('lower')),float(jo.get('upper'))])
        self.joint_limits = np.array(self.joint_limits)

    @property
    def available_paths(self):
        p = Paths()
        urdfs = [os.path.join(r,x) for r,d,y in os.walk(p.URDFS) for x in y if x.endswith('.urdf')]
        return urdfs

    @property
    def available_names(self):
        names = [os.path.basename(x).replace('.urdf','') for x in self.available_paths]
        return names

    @property
    def path(self):
        if self._get_path():
            return self.internal_path
        else:
            return None

    @path.setter
    def path(self, urdf_path):
        Paths().set('URDF',urdf_path)
        self._get_path()

    @property
    def name(self):
        if self._get_path():
            return os.path.basename(os.path.normpath(self.internal_path)).replace('.urdf','')
        else:
            return None
