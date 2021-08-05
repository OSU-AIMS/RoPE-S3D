# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley


import cv2
import numpy as np

from pixellib.instance import custom_segmentation

class RobotSegmenter():
    """LEGACY
    TODO: Deprecate
    Used to pre-segment the robot from the background"""

    def __init__(self, model_path):
        self.master = custom_segmentation()
        self.master.inferConfig(num_classes= 1, class_names= ["BG", "robot"])
        self.master.load_model(model_path)

    def segmentImage(self, img):
        # Load image if given path
        image = np.asarray(cv2.imread(img) if type(img) is str else img)

        tmp = np.copy(image)
        # Detect image
        r, output = self.master.segmentImage(tmp, process_frame=True)

        # Get mask
        mask = np.asarray(r['masks'])

        if mask.shape[2] == 0:
            # Make a fake mask
            mask = np.ones((tmp.shape[0:2]), dtype=bool)
        else:
            mask = mask[:,:,0]

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
                    to_go = int(round(look_up_dist * down**2 / (look_side_dist*1.5)**2))
                    # Truncate
                    if to_go > look_up_dist: to_go = look_up_dist
                    # Go down so many from row
                    mask[image.shape[0]-look_up_dist:image.shape[0]-look_up_dist+to_go,col] = True

        return mask
