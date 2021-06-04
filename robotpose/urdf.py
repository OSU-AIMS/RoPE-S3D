# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .CompactJSONEncoder import CompactJSONEncoder
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


    def guessDHParams(self):
        tree = ET.parse(self.internal_path)
        root = tree.getroot()

        origins = []
        axes = []
        for joint in root.findall('joint')[:6]:
            origins.append(str_to_list(joint.find('origin').get('xyz')))
            axes.append(str_to_list(joint.find('axis').get('xyz')))

        origins = np.array(origins)
        axes = np.array(axes)

        aa = np.zeros(7)
        alpha = np.zeros(7)
        dd = np.zeros(7)

        def complement(a):
            return a == 0
        def sign(a):
            return np.prod(a+complement(a))

        # Skip B, as it shares a location with R 
        for idx in range(1,5):
            aa[idx] = np.linalg.norm(complement(axes[idx-1])*origins[idx]) * sign(complement(axes[idx-1]))
            dd[idx] = np.linalg.norm(axes[idx-1]*origins[idx]) * sign(axes[idx-1])
        aa[6] = np.linalg.norm(complement(axes[5])*origins[5]) * sign(complement(axes[5]))
        dd[6] = np.linalg.norm(axes[5]*origins[5]) * sign(axes[5])

        """
        This could be entirely incorrect and inapplicable to other robots.
        May need to have user intervention.
        """
        def ang(a):
            a = a[0] + 1j*a[1]
            a = np.angle(a)
            if a < 0:
                a += 2*np.pi
            return a

        for idx in range(1,5):

            # Determine common normal axis of rotation
            axis = np.cross(axes[idx], axes[idx-1])

            # Convert 3D vectors to 2D vectors on plane of noraml axis
            new = axes[idx][axis==0]
            old = axes[idx-1][axis==0]

            # This is the sketchy part, but it works?
            mult = np.sum(axis) * np.sum(new+old) / np.abs(np.sum(new+old))

            if abs(sum(axis)) == 1:
                alpha[idx] = (ang(new) - ang(old)) * mult

        # Assume that the B and T alpha's are static (likely untrue)
        alpha[5] = -np.pi/2
        alpha[6] = np.pi

        params = {'a':aa,'alpha':alpha,'d':dd}

        if os.path.isfile(Paths().DH_PARAMS):
            with open(Paths().DH_PARAMS, 'w') as f:
                config = json.load(f)
        else:
            config = {}

        config[self.name] = params

        with open(Paths().DH_PARAMS, 'w') as f:
            f.write(CompactJSONEncoder(max_width=90,precise=True,indent=4).encode(config))

    def guessPoseConfig(self):
        tree = ET.parse(self.internal_path)
        root = tree.getroot()

        keys = self.mesh_names[:-1] # Don't include T
        config = {k:np.zeros(6) for k in keys}

        origins = []
        for joint in root.findall('joint')[:6]:
            origins.append(str_to_list(joint.find('origin').get('xyz')))
        origins = np.array(origins)

        # Base is normally shifted down
        config[keys[0]][2] = -1 * origins[0,2]

        # R is normally shifted back
        config[keys[4]][0] = -1 * origins[4,0]

        return config

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
