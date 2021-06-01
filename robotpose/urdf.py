import xml.etree.ElementTree as ET
import os
import sys
import numpy as np
from .paths import Paths

class URDFReader():
    def __init__(self):
        if self._get_path():
            self.load()

    def _get_path(self):
        p = Paths()
        if hasattr(p,'URDF'):
            self.path = p.URDF
            return True
        else:
            return False

    def load(self):
        tree = ET.parse(self.path)
        root = tree.getroot()
        self.meshes = []
        for link in root.findall('link')[:7]:
            self.meshes.append(link.find('visual').find('geometry').find('mesh').get('filename'))
        if sys.platform == 'win32':
            fileend = 'stl'
        elif sys.platform == 'linux':
            fileend = 'STL'
        self.meshes = [os.path.join('urdf',x.replace('package://',"").replace('STL',fileend)) for x in self.meshes]
        self.mesh_names = [os.path.splitext(os.path.basename(x))[0] for x in self.meshes]

        self.joint_limits = []
        for joint in root.findall('joint')[:6]:
            j = joint.find('limit')
            self.joint_limits.append([float(j.get('lower')),float(j.get('upper'))])
        self.joint_limits = np.array(self.joint_limits)


    @property
    def path(self):
        if self._get_path():
            return self.path
        else:
            return None

    @path.setter
    def path(self, urdf_path):
        Paths().set('URDF',urdf_path)
        self._get_path()

    @property
    def name(self):
        if self._get_path():
            return os.path.basename(os.path.normpath(self.path)).replace('.urdf','')
        else:
            return None
