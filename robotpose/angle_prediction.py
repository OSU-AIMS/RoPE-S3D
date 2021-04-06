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
        self.predictable_joints = [joint for joint in self.joint_data.keys() if len(self.joint_data[joint]['predictors']) > 0]

    def load(self, keypoint_detections, pointmap):
        self.detections = {}
        for name, idx in zip(self.keypoints, range(len(self.keypoints))):
            px = round(keypoint_detections[idx][0])
            py = round(keypoint_detections[idx][1])
            self.detections[name] = {'coords':pointmap[py,px], 'confidence':keypoint_detections[idx][2]}


    def predict(self):
        predictions = {}
        for joint in self.predictable_joints:
            predictions[joint] = self._predictAngle(joint)

        return predictions


    def _predictAngle(self,joint_name):
        tp = self.joint_data[joint_name]['type']

        if tp == 1:
            return self._type1predict(joint_name)
        elif tp == 2:
            return self._type2predict(joint_name)
        elif tp ==3 :
            return self._type3predict(joint_name)


    def _type1predict(self,joint_name):
        """ L, U, B Angles"""
        joint_info = self.joint_data[joint_name]

        pairs = []
        lengs = []
        confidence = []
        offsets = []
        for key in joint_info['predictors'].keys():
            points = [
                self.detections[joint_info['predictors'][key]['from']]['coords'],
                self.detections[joint_info['predictors'][key]['to']]['coords']
                ]
            pairs.append(points)
            lengs.append(joint_info['predictors'][key]['length'])
            conf = [
                self.detections[joint_info['predictors'][key]['to']]['confidence'],
                self.detections[joint_info['predictors'][key]['from']]['confidence']
                ]
            confidence.append(conf)
            offsets.append(joint_info['predictors'][key]['offset'])

        pairs = np.array(pairs)
        lengs = np.array(lengs)
        confidence = np.array(confidence)
        offsets = np.array(offsets)

        detected_lengs = self._distances(pairs)
        len_multipliers = np.exp(-(np.abs(lengs - detected_lengs)/lengs))
        print(len_multipliers)


    def _type2predict(self,joint_name):
        """ S Angle """
        pass


    def _type3predict(self,joint_name):
        """ R, T Angles """
        pass


    def _distances(self, point_arr):
        diff = point_arr[:,0,:] - point_arr[:,1,:]
        dist = np.power(np.sum(np.square(diff),-1),.5)
        return dist
