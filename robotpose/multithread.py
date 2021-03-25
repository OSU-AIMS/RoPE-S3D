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

    intrin = proj.makeIntrinsics()
    # Get pixel location of each point
    points_proj = proj.proj_point_to_pixel(intrin, points)

    points_proj_idx = np.zeros(points_proj.shape,dtype=int)
    points_proj_idx[:,0] = np.round(np.clip(points_proj[:,0],0,1279))
    points_proj_idx[:,1] = np.round(np.clip(points_proj[:,1],0,719))

    sum_arr = np.zeros((roi[2] - roi[0], roi[3] - roi[1], 3))
    count_arr = np.zeros((roi[2] - roi[0], roi[3] - roi[1], 3))

    for row in range(points_proj.shape[0]):
        if mask[points_proj_idx[row,1],points_proj_idx[row,0]]:
            # Shift based on ROI
            x = int(points_proj_idx[row,0] - roi[1])
            y = int(points_proj_idx[row,1] - roi[0])
            # Add to points
            sum_arr[y,x] += points[row]
            count_arr[y,x] += [1]*3
    
    count_arr[count_arr == 0] = 1   # Avoid dividing by 0

    ply_arr = sum_arr / count_arr

    mask_img = np.zeros((mask.shape[0],mask.shape[1],3))
    for idx in range(3):
        mask_img[:,:,idx] = mask
    output_image = np.multiply(image, mask_img).astype(np.uint8)
    output_image = output_image[roi[0]:roi[2],roi[1]:roi[3]]


    return output_image, ply_arr








# def generateMap(points, intrin_type = '1280_720_color'):
#     intrin = proj.makeIntrinsics(intrin_type)

#     # Instead of an RGB/BGR array, this is an XYZ array
#     sum_arr = np.zeros((intrin.height, intrin.width, 3))
#     count_arr = np.zeros((intrin.height, intrin.width, 3))

#     points_idx = np.array(np.round(proj.proj_point_to_pixel(intrin, points)), dtype=int)

#     points[:,2] *= -1

#     points_idx[:,0] = np.round(np.clip(points_idx[:,0],0,intrin.width-1))
#     points_idx[:,1] = np.round(np.clip(points_idx[:,1],0,intrin.height-1))

#     for pixel, loc in zip(points_idx,points):
#         px, py = pixel
#         sum_arr[py,px] += loc
#         count_arr[py,px] += [1]*3

#     arr = sum_arr / count_arr
#     arr = np.nan_to_num(arr,nan=0,posinf=0,neginf=0)

#     return arr





def smoothMap2(map):
    sum_arr = np.zeros(map.shape)
    count_arr = np.copy(sum_arr)

    for radius in range(2):
        weight = .25 ** radius
        for r in range(radius,map.shape[0]-radius):
            for c in range(radius,map.shape[1]-radius):
                if np.any(map[r,c]):
                    sum_arr[r-radius:r+radius,c-radius:c+radius] += map[r,c] * weight
                    count_arr[r-radius:r+radius,c-radius:c+radius] += [weight] * 3

    arr = sum_arr / count_arr
    arr = np.nan_to_num(arr,nan=0,posinf=0,neginf=0)
    return arr