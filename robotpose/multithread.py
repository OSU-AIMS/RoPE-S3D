# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import open3d as o3d
import numpy as np
from . import projection as proj
from .utils import expandRegion


def crop(depthmap, image, mask, roi):

    # Convert depthmap to pointmap
    intrin = proj.makeIntrinsics()
    pointmap = proj.deproj_depthmap_to_pointmap(intrin, depthmap)
    #pointmap = proj.deproj_depthmap_to_pointmap_different(proj.makeIntrinsics('1280_720_depth'),proj.makeIntrinsics(), depthmap)
    #pointmap = proj.deproj_depthmap_to_pointmap(intrin, depthmap[roi[0]:roi[2],roi[1]:roi[3]], x_offset=roi[1], y_offset=roi[0])

    mask_img = np.zeros((mask.shape[0],mask.shape[1],3))
    for idx in range(3):
        mask_img[:,:,idx] = mask
    output_image = np.multiply(image, expandRegion(mask_img,25)).astype(np.uint8)
    output_image = output_image[roi[0]:roi[2],roi[1]:roi[3]]
    pointmap = np.multiply(pointmap, mask_img)
    pointmap = pointmap[roi[0]:roi[2],roi[1]:roi[3]]

    return output_image, pointmap


