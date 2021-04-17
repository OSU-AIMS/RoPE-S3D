import xml.etree.ElementTree as ET
import os
import json
from .CompactJSONEncoder import CompactJSONEncoder
import numpy as np

JSON_PATH = r'data/paths.json'

class URDFReader():
    def __init__(self):
        if self.get_path():
            self.load()

    def store_path(self, urdf_path):
        if os.path.isfile(JSON_PATH):
            with open(JSON_PATH,'r') as f:
                data = json.load(f)
        else:
            data = {}
        data['urdf_path'] = urdf_path
        with open(JSON_PATH,'w') as f:
            f.write(CompactJSONEncoder(indent=4).encode(data))

    def get_path(self):
        if os.path.isfile(JSON_PATH):
            with open(JSON_PATH,'r') as f:
                data = json.load(f)
            if 'urdf_path' in data.keys():
                self.path = data['urdf_path']
                return True
            else:
                return False
        else:
            return False

    def return_path(self):
        if self.get_path():
            return self.path
        else:
            return None

    def load(self):
        tree = ET.parse(self.path)
        root = tree.getroot()
        self.meshes = []
        for link in root.findall('link'):
            self.meshes.append(link.find('visual').find('geometry').find('mesh').get('filename'))
        self.meshes = [os.path.join('urdf',x.replace('package://',"").replace('STL','stl')) for x in self.meshes]

        self.joint_limits = []
        for joint in root.findall('joint'):
            j = joint.find('limit')
            self.joint_limits.append([float(j.get('lower')),float(j.get('upper'))])
        self.joint_limits = np.array(self.joint_limits)
