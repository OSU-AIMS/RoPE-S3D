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
import sys
import csv

from . import paths as p


class Skeleton():

    def __init__(self, name):
        self.name = name

        csv_ = name + '.csv' in os.listdir(p.SKELETONS)
        json_ = name + '.json' in os.listdir(p.SKELETONS)

        if not csv_:
            raise ValueError(
                f"The skeleton base document, {name + '.csv'} was not found in {p.SKELETONS}.\
                Please create a skeleton before attempting to use it.")

        self.csv_path = os.path.join(p.SKELETONS, name + '.csv')

        if json_:
            self.json_path = os.path.join(p.SKELETONS, name + '.json')
        else:
            self._makeJSON()

        self.update()


    def update(self):
        with open(self.json_path,'r') as f:
            self.data = json.load(f)

        self.keypoints = [x for x in self.data['keypoints'].keys()]
        self.keypoint_data = self.data['keypoints']
        try:
            self.joint_data = self.data['joints']
        except KeyError:
            print("Skeleton Joint Config Missing")

    def _hasJointConfig(self):
        return 'joints' in self.data.keys()


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
        
        json_info = {}
        json_info['markers'] = {"height": 0.005,"radius": 0.005}

        default_keypoint_entry = {"parent_joint": "joint_name","pose":[0.1,0,0,1.5707963267948966,0,0]}
        keypoint_data = {}
        for keypoint in keypoints:
            keypoint_data[keypoint] = default_keypoint_entry
        json_info['keypoints'] = keypoint_data

        default_predictor_entry = {"from": "keypoint","to": "another_keypoint","length": 1.0,"offset":0}
        default_predictors = {"A":default_predictor_entry,"B":default_predictor_entry}
        joint_angle_data = {}
        joint_angle_data['S'] = {"type":2,"max":2,"min":-2,"parent":None,"parent_mult":0,"parent_offset":0,"self_mult":1,"predictors":default_predictors}
        joint_angle_data['L'] = {"type":1,"max":4,"min":-4,"parent":None,"parent_mult":0,"parent_offset":0,"self_mult":1,"predictors":default_predictors}
        joint_angle_data['U'] = {"type":1,"max":4,"min":-4,"parent":'L',"parent_mult":1,"parent_offset":0,"self_mult":1,"predictors":default_predictors}
        #joint_angle_data['R'] = {"type":3,"max":2,"min":-2,"predictors":default_predictors}
        joint_angle_data['R'] = {"type":3,"max":2,"min":-2,"parent":None,"parent_mult":0,"parent_offset":0,"self_mult":1,"predictors":{}}
        joint_angle_data['B'] = {"type":1,"max":4,"min":-4,"parent":'U',"parent_mult":1,"parent_offset":0,"self_mult":1,"predictors":default_predictors}
        #joint_angle_data['T'] = {"type":3,"max":2,"min":-2,"predictors":default_predictors}
        joint_angle_data['T'] = {"type":3,"max":2,"min":-2,"parent":None,"parent_mult":0,"parent_offset":0,"self_mult":1,"predictors":{}}

        json_info['joints'] = joint_angle_data

        with open(self.csv_path.replace('.csv','.json'),'w') as f:
            f.write(self._removePoseIndent(json.dumps(json_info, indent=4)))

        sys.exit(f"\nMade JSON configuration for skeleton {self.name}\nPlease configure before using.\n")


    def _removePoseIndent(self, string):
        return string.replace(
            '"pose": [\n                0.1,\n                0,\n                0,\n                1.5707963267948966,\n                0,\n                0\n            ]',
            '"pose": [0.1, 0, 0, 1.5707963267948966, 0, 0]')