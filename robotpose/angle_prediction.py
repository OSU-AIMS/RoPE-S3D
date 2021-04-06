# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from .skeleton import Skeleton
import numpy as np


class Predictor(Skeleton):
    
    def __init__(self, skele_name):
        super().__init__(skele_name)
        assert self._hasJointConfig(),"Joint Configuration must be present for angle prediction."

    def load(self, keypoint_detections, pointmap):
        self.detections = {}
        for name, idx in zip(self.keypoints, range(len(self.keypoints))):
            px = round(keypoint_detections[idx][0])
            py = round(keypoint_detections[idx][1])
            self.detections[name] = {'coords':pointmap[py,px], 'confidence':keypoint_detections[idx][2]}

    def predict(self):
        predictions = {}
        for joint in self.joint_data.keys():
            predictions[joint] = self._predictAngle(joint)

        return predictions

    def _predictAngle(self,joint_name):
        type_switch = {
            1:self._type1predict(joint_name),
            2:self._type2predict(joint_name),
            3:self._type3predict(joint_name)
                    }

        return type_switch[self.joint_data[joint_name]['type']]()



    def _type1predict(self,joint_name):
        """ L, U, B Angles"""
        joint_info = self.joint_data[joint_name]

        pairs = []
        weights = []
        offsets = []
        for key in joint_info['predictors'].keys():
            points = [self.detections[joint_info['predictors'][key]['from']]['coords'],
                self.detections[joint_info['predictors'][key]['to']]['coords']]
            pairs.append(points)
            weighters = [joint_info['predictors'][key]['length'], self.detections[joint_info['predictors'][key]['to']]['confidence']]
            weights.append(weighters)
            offsets.append(joint_info['predictors'][key]['offset'])

        pairs = np.array(pairs)
        weights = np.array(weights)
        offsets = np.array(offsets)




    def _type2predict(self,joint_name):
        """ S Angle """
        pass

    def _type3predict(self,joint_name):
        """ R, T Angles """
        pass