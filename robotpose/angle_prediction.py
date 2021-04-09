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
            estimated = False
            if not np.any(coords):
                coords = fill_hole(pointmap, py, px, 50)
                estimated = True

            self.detections[name] = {'coords':coords, 'confidence':keypoint_detections[idx][2], 'estimated':estimated}


    def predict(self):
        predictions = {}
        for joint in self.predictable_joints:
            preds, est = self._predictAngle(joint)
            predictions[joint] = {"val": preds,"percent_est":est}

        return predictions


    def _predictAngle(self,joint_name):
        tp = self.joint_data[joint_name]['type']

        if tp == 1:
            pred, est = self._type1predict(joint_name)
        elif tp == 2:
            pred, est = self._type2predict(joint_name)
        elif tp == 3:
            pred, est = self._type3predict(joint_name)

        if pred is None:
            return None, 1

        pred += self.joint_data[joint_name]['parent_offset']

        while pred < self.joint_data[joint_name]['min']:
            pred += 2 * np.pi
        
        while pred > self.joint_data[joint_name]['max']:
            pred -= 2 * np.pi

        return pred, est


    def _type1predict(self,joint_name):
        """ L, U, B Angles"""
        pairs, lengs, confidence, offsets, estimate = self._getPredictors(joint_name)
        multipliers = self._type1Multipliers(pairs, lengs, confidence, estimate)
        
        vecs = pairs[:,1] - pairs[:,0]
        rot_y = vecs[:,1]
        rot_x = np.sqrt(vecs[:,0] ** 2 + vecs[:,2] ** 2) * abs(vecs[:,0]) / vecs[:,0]

        # Convert to angles between +/-pi rad
        preds = np.arctan(rot_y/rot_x)
        neg_x = rot_x < 0
        neg_y = rot_y < 0
        pos_y = rot_y > 0

        preds[np.where(neg_x & pos_y)] = preds[np.where(neg_x & pos_y)] + np.pi
        preds[np.where(neg_x & neg_y)] = -np.pi/2 - preds[np.where(neg_x & neg_y)]

        preds += offsets

        # Range discontinuity around +pi and -pi. Convert to positive angles to see if stdev is lower.
        # if len(preds) > 1:
        #     base_std = np.std(preds)
        #     preds_alt = np.copy(preds)
        #     preds_alt[np.where(preds_alt < 0)] = 2*np.pi + preds_alt[np.where(preds_alt < 0)]
        #     alt_std = np.std(preds_alt)
        #     if alt_std < base_std:
        #         preds = preds_alt

        pred = np.sum(multipliers * preds) / np.sum(multipliers)
    
        return pred, sum(estimate) / len(estimate)


    def _type2predict(self,joint_name):
        """ S Angle """
        pairs, lengs, confidence, offsets, estimate = self._getPredictors(joint_name)
    
        vecs = pairs[:,1] - pairs[:,0]
        preds = np.arctan(vecs[:,2]/vecs[:,0])

        multipliers = self._type2Multipliers(pairs, lengs, confidence,estimate, vecs)

        preds += offsets
        pred = np.sum(multipliers * preds) / np.sum(multipliers)

        return pred, sum(estimate) / len(estimate)


    def _type3predict(self,joint_name):
        """ R, T Angles """
        pass


    def _type2Multipliers(self,pairs,lengs,confidence,estimate,vecs):
        detected_lengs = self._distances(pairs)
        len_multipliers = np.exp(-(np.abs(lengs - detected_lengs)/lengs))
        confidence_multipliers = 2*lengs *(np.power(np.prod(confidence,-1),1.75) / np.square(np.sum(confidence,-1)))
        x_distance_multiplier = vecs[:,0] / detected_lengs
        est_multipliers = 1 - .5 * estimate
        multipliers = len_multipliers * confidence_multipliers * x_distance_multiplier * est_multipliers
        return multipliers


    def _type1Multipliers(self,pairs,lengs,confidence, estimate):
        detected_lengs = self._distances(pairs)
        len_multipliers = np.exp(-(np.abs(lengs - detected_lengs)/lengs))
        confidence_multipliers = 2*lengs *(np.power(np.prod(confidence,-1),1.75) / np.square(np.sum(confidence,-1)))
        est_multipliers = 1 - .5 * estimate
        multipliers = len_multipliers * confidence_multipliers * est_multipliers
        return multipliers


    def _getPredictors(self, joint_name):
        joint_info = self.joint_data[joint_name]

        pairs = []
        lengs = []
        confidence = []
        offsets = []
        contains_estimate = []
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
            contains_estimate.append(
                self.detections[joint_info['predictors'][key]['to']]['estimated'] or
                self.detections[joint_info['predictors'][key]['from']]['estimated']
                )

        pairs = np.array(pairs)
        lengs = np.array(lengs)
        confidence = np.array(confidence)
        offsets = np.array(offsets)
        contains_estimate = np.array(contains_estimate)
        return pairs, lengs, confidence, offsets, contains_estimate


    def _distances(self, point_arr):
        diff = point_arr[:,0,:] - point_arr[:,1,:]
        dist = np.power(np.sum(np.square(diff),-1),.5)
        return dist
