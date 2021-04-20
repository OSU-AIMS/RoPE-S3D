# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import json
import numpy as np
import os
import csv
import string

from .paths import Paths as p
from .CompactJSONEncoder import CompactJSONEncoder
from .urdf import URDFReader

DEFAULT_CSV = "name,parent,swap\nbase,,\nL,base,\nmidL,L,\nU,midL,\npreR,U,\nR,preR,\nB,R,\nT,B,\n"



class SkeletonInfo:
    
    def __init__(self):
        pass

    def valid(self):
        return [x.replace('.csv','') for x in os.listdir(p().SKELETONS) if x.endswith('.csv') and os.path.isfile(os.path.join(p().SKELETONS,x.replace('.csv','.json')))]

    def incomplete(self):
        return [x.replace('.csv','') for x in os.listdir(p().SKELETONS) if x.endswith(".csv") and x.replace('.csv','') not in self.valid()]

    def num_incomplete(self):
        return len([x for x in os.listdir(p().SKELETONS) if x.endswith(".csv")]) - len(self.valid())

    def create_csv(self,name):
        with open(os.path.join(p().SKELETONS,f"{name}.csv"), 'w') as f:
            f.write(DEFAULT_CSV)
        return os.path.join(p().SKELETONS,f"{name}.csv")



class Skeleton():

    def __init__(self, name, create = False):
        if name is None:
            name = 'BASE'
        self.name = name

        csv_ = name + '.csv' in os.listdir(p().SKELETONS)
        json_ = name + '.json' in os.listdir(p().SKELETONS)

        if not csv_:
            raise ValueError(
                f"The skeleton base document, {name + '.csv'} was not found in {p().SKELETONS}."+
                "Please create a skeleton before attempting to use it.")

        self.csv_path = os.path.join(p().SKELETONS, name + '.csv')

        if json_:
            self.json_path = os.path.join(p().SKELETONS, name + '.json')
        elif create:
            self._makeJSON()
        else:
            raise ValueError(
                f"The skeleton JSON document, {name + '.json'} was not found in {p().SKELETONS}"+
                "And the skeleton was not created with the intent of making a new JSON.\n"+
                "To create a JSON, call Skeleton with create = True.")

        self.update()


    def update(self):
        with open(self.json_path,'r') as f:
            self.data = json.load(f)

        self.keypoints = [x for x in self.data['keypoints'].keys()]
        self.keypoint_data = self.data['keypoints']
        self.joint_data = self.data['joints']


    def _writeJSON(self):
        assert hasattr(self, 'data'), "Data must exist to write"
        while True:
            try:
                with open(self.json_path,'w') as f:
                    f.write(CompactJSONEncoder(indent=4).encode(self.data))
                break
            except PermissionError:
                pass
        self.update()

    def _hasJointConfig(self):
        return 'joints' in self.data.keys()

    def _addKeypoint_csv(self, keypoint):
        with open(self.csv_path,'a') as f:
            f.write(f"{keypoint},,\n")

    def _addKeypoint_json(self, keypoint):
        self.data['keypoints'][keypoint] = {"parent_keypoint": None,"parent_link": "link_name","pose":[0.1,0,0,1.570796,0,1.570796]}
        self._writeJSON()

    def _changeParent_csv(self, keypoint, parent):
        assert keypoint in self.keypoints, "Keypoint must be in skeleton to edit"
        with open(self.csv_path, 'r') as f:
            dat = ''
            while not dat.startswith(keypoint):
                dat = f.readline()
            replace = dat
            f.seek(0)
            full = f.read()
        with open(self.csv_path, 'w') as f:
            f.write(full.replace(replace,f"{keypoint},{parent},\n"))

    def _changeParentPoint_json(self, keypoint, parent):
        if parent == '':
            parent = None
        self.data['keypoints'][keypoint]['parent_keypoint'] = parent
        self._writeJSON()

    def _changeParentLink_json(self, keypoint, parent):
        self.data['keypoints'][keypoint]['parent_link'] = parent
        self._writeJSON()
        
    def _removeKeypoint_csv(self, keypoint):
        assert keypoint in self.keypoints, "Keypoint must be in skeleton to remove"
        with open(self.csv_path, 'r') as f:
            dat = ''
            while not dat.startswith(keypoint):
                dat = f.readline()
            replace = dat
            f.seek(0)
            full = f.read()
        with open(self.csv_path, 'w') as f:
            f.write(full.replace(replace,''))

    def _removeKeypoint_json(self, keypoint):
        del self.data['keypoints'][keypoint]

        # Remove from all predictors
        for joint in ['S','L','U','R','B']:
            joint_data = self.data['joints'][joint]
            for pred in self.data['joints'][joint]['predictors']:
                if joint_data['predictors'][pred]['from'] == keypoint or joint_data['predictors'][pred]['to'] == keypoint:
                    del self.data['joints'][joint]['predictors'][pred]

        self._writeJSON()

    def _renameKeypoint_csv(self, past_name, new_name):
        assert past_name in self.keypoints, "Keypoint must be in skeleton to rename"
        with open(self.csv_path, 'r') as f:
            full = f.read()
        full = full.replace(f"\n{past_name},",f"\n{new_name},")
        full = full.replace(f",{past_name},",f",{new_name},")
        with open(self.csv_path, 'w') as f:
            f.write(full)

    def _renameKeypoint_json(self, past_name, new_name):
        self.data['keypoints'][new_name] = self.data['keypoints'][past_name]
        del self.data['keypoints'][past_name]

        for joint in ['S','L','U','R','B']:
            joint_data = self.data['joints'][joint]
            for pred in self.data['joints'][joint]['predictors']:
                if joint_data['predictors'][pred]['from'] == past_name:
                    self.data['joints'][joint]['predictors'][pred]['from'] = new_name
                if joint_data['predictors'][pred]['to'] == past_name:
                    self.data['joints'][joint]['predictors'][pred]['from'] = new_name

        self._writeJSON()


    def _addKeypoint(self, keypoint):
        self.update()
        self._addKeypoint_csv(keypoint)
        self._addKeypoint_json(keypoint)

    def _changeKeypointParentLink(self, keypoint, parent):
        self.update()
        self._changeParentLink_json(keypoint, parent)

    def _changeKeypointParentPoint(self, keypoint, parent):
        self._changeParent_csv(keypoint, parent)
        self._changeParentPoint_json(keypoint, parent)

    def _removeKeypoint(self, keypoint):
        self.update()
        self._removeKeypoint_csv(keypoint)
        self._removeKeypoint_json(keypoint)

    def _renameKeypoint(self, old_name, new_name):
        assert old_name in self.keypoints, "Old name must be in keypoints"
        assert new_name not in self.keypoints, "Cannot change to already-occupied name"
        self.update()
        self._renameKeypoint_csv(old_name, new_name)
        self._renameKeypoint_json(old_name, new_name)

    def _changeKeypointPose(self, keypoint, pose):
        self.data['keypoints'][keypoint]['pose'] = pose
        self._writeJSON()

    def _addPredictor(self, joint):
        names = list(string.ascii_uppercase)
        names = [x for x in names if x not in [y for y in self.joint_data[joint]['predictors'].keys()]]
        name = names[0]
        default_predictor_entry = {"from": "","to": "","length": 0,"offset":0}
        self.data['joints'][joint]['predictors'][name] = default_predictor_entry
        self._writeJSON()

    def _removePredictor(self, joint, predictor):
        del self.data['joints'][joint]['predictors'][predictor]
        self._writeJSON()

        alphabet = list(string.ascii_uppercase)
        to_cycle = [x for x in alphabet if alphabet.index(x) > alphabet.index(predictor) and x in self.joint_data[joint]['predictors'].keys()]
            
        for key in to_cycle:
            new = alphabet[alphabet.index(key) - 1]
            self.data['joints'][joint]['predictors'][new] = self.data['joints'][joint]['predictors'][key]
            del self.data['joints'][joint]['predictors'][key]
        self._writeJSON()




    def _makeJSON(self):
        
        with open(self.csv_path, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            csv_data = []
            for row in reader:
                csv_data.append(row)
        csv_data = np.array(csv_data)
        csv_data = csv_data[1:]
        csv_data = csv_data[:,:-1]
        keypoints = csv_data[:,0]
        parents = csv_data[:,1]
        
        json_info = {}
        json_info['markers'] = {"height": 0.005,"radius": 0.005}

        keypoint_data = {}
        for keypoint, parent in zip(keypoints,parents):
            if parent == '':
                parent = None
            keypoint_data[keypoint] = {"parent_keypoint": parent,"parent_link": "link_name","pose":[0.1,0,0,1.570796,0,0]}
        json_info['keypoints'] = keypoint_data

        u_reader = URDFReader()
        lims = u_reader.joint_limits

        default_predictor_entry = {"from": "keypoint","to": "another_keypoint","length": 0,"offset":0}
        default_predictors = {"A":default_predictor_entry,"B":default_predictor_entry}
        joint_angle_data = {}
        joint_angle_data['S'] = {"type":2,"max":lims[0,1],"min":lims[0,0],"parent":None,"parent_mult":0,"offset":0,"self_mult":1,"predictors":default_predictors}
        joint_angle_data['L'] = {"type":1,"max":lims[1,1],"min":lims[1,0],"parent":None,"parent_mult":0,"offset":-np.pi/2,"self_mult":-1,"predictors":default_predictors}
        joint_angle_data['U'] = {"type":1,"max":lims[2,1],"min":lims[2,0],"parent":'L',"parent_mult":1,"offset":0,"self_mult":1,"predictors":default_predictors}
        joint_angle_data['R'] = {"type":3,"max":lims[3,1],"min":lims[3,0],"parent":None,"parent_mult":0,"offset":0,"self_mult":1,"predictors":{}}
        joint_angle_data['B'] = {"type":1,"max":lims[4,1],"min":lims[4,0],"parent":'U',"parent_mult":1,"offset":0,"self_mult":1,"predictors":default_predictors}
        joint_angle_data['T'] = {"type":3,"max":lims[5,1],"min":lims[5,0],"parent":None,"parent_mult":0,"offset":0,"self_mult":1,"predictors":{}}

        json_info['joints'] = joint_angle_data

        self.json_path = self.csv_path.replace('.csv','.json')
        with open(self.json_path,'w') as f:
            f.write(CompactJSONEncoder(indent=4).encode(json_info))

        print(f"\nMade JSON configuration for skeleton {self.name}\nPlease configure before using.\n")
