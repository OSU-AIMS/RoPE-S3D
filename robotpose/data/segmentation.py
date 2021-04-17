# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley


import pixellib
from pixellib.instance import custom_segmentation
import cv2
import os
import numpy as np
import pyrealsense2 as rs
from ..paths import Paths as p
from .. import projection as proj


class RobotSegmenter():

    def __init__(self, resolution = (720,800), model_path = os.path.join(p().SEG_MODELS,'D.h5'), intrinsics = '1280_720_color'):
        self.master = custom_segmentation()
        self.master.inferConfig(num_classes= 1, class_names= ["BG", "mh5"])
        self.master.load_model(model_path)
        self.crop_resolution = resolution
        self.intrinsics = proj.makeIntrinsics(intrinsics)

    def height(self):
        return self.crop_resolution[0]
    def width(self):
        return self.crop_resolution[1]

    def segmentImage(self, img):
        # Load image if given path
        if type(img) is str:
            image = cv2.imread(img)
        else:
            image = img

        image = np.asarray(image)
        tmp = np.copy(image)
        # Detect image
        r, output = self.master.segmentImage(tmp, process_frame=True)

        # Get mask and roi
        mask = np.asarray(r['masks'])

        if mask.shape[2] == 0:
            # Make a fake mask
            mask = np.ones((tmp.shape[0:2]), dtype=bool)
            height, width = tmp.shape[0:2]
            roi = [
                (height-self.crop_resolution[0])/2,
                (width-self.crop_resolution[1])/2,
                (height-self.crop_resolution[0])/2 + self.crop_resolution[0],
                (width-self.crop_resolution[1])/2 + self.crop_resolution[1],
                ]
        else:
            roi = r['rois'][0] # Y1,X1,Y2,X2
            mask = mask[:,:,0]

            # Expand ROI up and down
            while roi[2] - roi[0] < self.crop_resolution[0]:
                # Expand Up
                if roi[0] > 0:
                    roi[0] -= 1
                #Expand Down
                if roi[2] < image.shape[0]:
                    roi[2] += 1

            # Make sure ROI is exact crop size needed
            while roi[2] - roi[0] > self.crop_resolution[0]:
                roi[0] += 1

            # Expand ROI left and right
            while roi[3] - roi[1] < self.crop_resolution[1]:
                # Expand Left
                if roi[1] > 0:
                    roi[1] -= 1
                #Expand Right
                if roi[3] < image.shape[1]:
                    roi[3] += 1

            # Make sure ROI is exact crop size needed
            while roi[3] - roi[1] > self.crop_resolution[1]:
                roi[1] += 1

        assert roi[3] - roi[1] == self.crop_resolution[1], f"ROI Crop Width Incorrect. {roi[3] - roi[1]} != {self.crop_resolution[1]}"
        assert roi[2] - roi[0] == self.crop_resolution[0], f"ROI Crop Height Incorrect.{roi[2] - roi[0]} != {self.crop_resolution[0]}"


        """
        Mask Modifications

        Usually doesn't segment ~20 pix from bottom
        """

        # Base how far it goes down on how many are around it in an x-pixel radius
        look_up_dist = 27
        look_side_dist = 10 # one way

        for col in range(mask.shape[1]):
                if mask[image.shape[0]-look_up_dist,col]:
                    # Find how many are each side
                    down = np.sum(mask[image.shape[0]-look_up_dist,col-look_side_dist:col+look_side_dist])
                    # Arbitrary calc
                    to_go = round(look_up_dist * down**2 / (look_side_dist*1.5)**2)
                    # Truncate
                    if to_go > look_up_dist:
                        to_go = look_up_dist
                    # Go down so many from row
                    mask[image.shape[0]-look_up_dist:image.shape[0]-look_up_dist+to_go,col] = True

        return mask, roi
