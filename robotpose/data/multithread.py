# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import numpy as np
from .. import projection as proj
from ..utils import expandRegion


def crop(depthmap, image, mask):

    # Convert depthmap to pointmap
    intrin = proj.makePresetIntrinsics()
    pointmap = proj.deproj_depthmap_to_pointmap(intrin, depthmap)

    mask_img = np.zeros((mask.shape[0],mask.shape[1],3))
    for idx in range(3):
        mask_img[:,:,idx] = mask
    output_image = np.multiply(image, expandRegion(mask_img,10)).astype(np.uint8)
    pointmap = np.multiply(pointmap, mask_img)

    return output_image, pointmap


