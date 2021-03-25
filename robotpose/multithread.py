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


def crop(ply_path, image, mask, roi):

    # Open PLY
    cloud = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(cloud.points)

    # Invert X Coords
    points[:,0] = points[:,0] * -1

    # Align XYZ points relative to the color camera instead of the depth camera
    points[:,0] -= .0175

    crop_ply_data = []

    intrinsics = proj.makeIntrinsics()
    # Get pixel location of each point
    points_proj = proj.proj_point_to_pixel(intrinsics, points)

    points_proj_idx = np.zeros(points_proj.shape,dtype=int)
    points_proj_idx[:,0] = np.round(np.clip(points_proj[:,0],0,1279))
    points_proj_idx[:,1] = np.round(np.clip(points_proj[:,1],0,719))

    for row in range(points_proj.shape[0]):
        if mask[points_proj_idx[row,1],points_proj_idx[row,0]]:
            # Shift based on ROI
            points_proj[row,0] -= roi[1]
            points_proj[row,1] -= roi[0]
            crop_ply_data.append(np.append(points_proj[row,:], points[row,:]))

    mask_img = np.zeros((mask.shape[0],mask.shape[1],3))
    for idx in range(3):
        mask_img[:,:,idx] = mask
    output_image = np.multiply(image, mask_img).astype(np.uint8)
    output_image = output_image[roi[0]:roi[2],roi[1]:roi[3]]


    return output_image, crop_ply_data