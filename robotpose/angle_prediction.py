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
from .projection import fill_hole


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

            coords = pointmap[py,px]

            if not np.any(coords):
                coords = fill_hole(pointmap, py, px, 50)

            self.detections[name] = {'coords':coords, 'confidence':keypoint_detections[idx][2]}


    def predict(self):
        predictions = {}
        for joint in self.predictable_joints:
            predictions[joint] = self._predictAngle(joint)

        return predictions


    def _predictAngle(self,joint_name):
        tp = self.joint_data[joint_name]['type']

        if tp == 1:
            pred = self._type1predict(joint_name)
        elif tp == 2:
            pred = self._type2predict(joint_name)
        elif tp == 3:
            pred = self._type3predict(joint_name)

        if pred is None:
            return None

        pred += self.joint_data[joint_name]['parent_offset']

        if pred < self.joint_data[joint_name]['min']:
            pred += 2 * np.pi
        
        if pred > self.joint_data[joint_name]['max']:
            pred -= 2 * np.pi

        return pred


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
        confidence_multipliers = 2*lengs *(np.power(np.prod(confidence,-1),1.75) / np.square(np.sum(confidence,-1)))
        multipliers = len_multipliers * confidence_multipliers

        
        vecs = pairs[:,1] - pairs[:,0]
        rot_y = vecs[:,1]
        rot_x = np.sqrt(vecs[:,0] ** 2 + vecs[:,2] ** 2) * abs(vecs[:,0]) / vecs[:,0]

        if joint_name == 'L':
            print(vecs)


        # Convert to angles between +/-pi rad
        preds = np.arctan(rot_y/rot_x)
        neg_x = rot_x < 0
        neg_y = rot_y < 0
        pos_y = rot_y > 0

        preds[np.where(neg_x & pos_y)] = preds[np.where(neg_x & pos_y)] + np.pi
        preds[np.where(neg_x & neg_y)] = -np.pi/2 - preds[np.where(neg_x & neg_y)]

        preds += offsets

        # Range discontinuity around +pi and -pi. Convert to positive angles to see if stdev is lower.
        if len(preds) > 1:
            base_std = np.std(preds)
            preds_alt = np.copy(preds)
            preds_alt[np.where(preds_alt < 0)] = 2*np.pi + preds_alt[np.where(preds_alt < 0)]
            alt_std = np.std(preds_alt)
            if alt_std < base_std:
                preds = preds_alt

        pred = np.sum(multipliers * preds) / np.sum(multipliers)

    
        return pred


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
