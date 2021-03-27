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


# def crop(ply_path, image, mask, roi):

#     # Open PLY
#     cloud = o3d.io.read_point_cloud(ply_path)
#     points = np.asarray(cloud.points)

#     # Invert X Coords
#     points[:,0] = points[:,0] * -1

#     # Align XYZ points relative to the color camera instead of the depth camera
#     points[:,0] -= .0175

#     # Get pixel location of each point
#     intrin = proj.makeIntrinsics()
#     points_proj = proj.proj_point_to_pixel(intrin, points)

#     points_proj[:,0] = np.round(np.clip(points_proj[:,0],0,1279))
#     points_proj[:,1] = np.round(np.clip(points_proj[:,1],0,719))
#     points_proj = np.array(points_proj, dtype=int)

#     sum_arr = np.zeros((roi[2] - roi[0], roi[3] - roi[1], 3))
#     count_arr = np.zeros((roi[2] - roi[0], roi[3] - roi[1], 3))

#     for row in range(points_proj.shape[0]):
#         if mask[points_proj[row,1],points_proj[row,0]]:
#             # Shift based on ROI
#             x = int(points_proj[row,0] - roi[1])
#             y = int(points_proj[row,1] - roi[0])
#             # Add to points
#             sum_arr[y,x] += points[row]
#             count_arr[y,x] += [1]*3

    
#     count_arr[count_arr == 0] = 1   # Avoid dividing by 0

#     ply_arr = sum_arr / count_arr

#     #ply_arr = smoothMapMask(ply_arr,mask,roi)

#     mask_img = np.zeros((mask.shape[0],mask.shape[1],3))
#     for idx in range(3):
#         mask_img[:,:,idx] = mask
#     output_image = np.multiply(image, mask_img).astype(np.uint8)
#     output_image = output_image[roi[0]:roi[2],roi[1]:roi[3]]


#     return output_image, ply_arr


def crop(depthmap, image, mask, roi):

    # Convert depthmap to pointmap
    intrin = proj.makeIntrinsics()
    pointmap = proj.deproj_depthmap_to_pointmap(intrin, depthmap)
    #pointmap = proj.deproj_depthmap_to_pointmap(intrin, depthmap[roi[0]:roi[2],roi[1]:roi[3]], x_offset=roi[1], y_offset=roi[0])

    mask_img = np.zeros((mask.shape[0],mask.shape[1],3))
    for idx in range(3):
        mask_img[:,:,idx] = mask
    output_image = np.multiply(image, mask_img).astype(np.uint8)
    output_image = output_image[roi[0]:roi[2],roi[1]:roi[3]]

    return output_image, pointmap






# def smoothMap(map, mask, roi):
#     mask = mask[roi[0]:roi[2],roi[1]:roi[3]]
#     sum_arr = np.zeros(map.shape)
#     count_arr = np.copy(sum_arr)

#     for radius in range(2):
#         weight = .25 ** radius
#         for r in range(radius,map.shape[0]-radius):
#             for c in range(radius,map.shape[1]-radius):
#                 if np.any(map[r,c]):
#                     sum_arr[r-radius:r+radius,c-radius:c+radius] += map[r,c] * weight
#                     count_arr[r-radius:r+radius,c-radius:c+radius] += [weight] * 3

#     arr = sum_arr / count_arr
#     arr = np.nan_to_num(arr,nan=0,posinf=0,neginf=0)
#     return arr



# def smoothMapMask(map, mask, roi):
#     mask = mask[roi[0]:roi[2],roi[1]:roi[3]]
#     sum_arr = np.zeros(map.shape)
#     count_arr = np.copy(sum_arr)

#     radius = 0
#     while np.min(count_arr[np.where(mask == True)]) < 1:

#         weight = .25 ** radius

#         rc_min = radius
#         r_max = map.shape[0] - radius - 1
#         c_max = map.shape[1] - radius - 1

#         idx_arr = np.array(np.where(np.any(map,-1)), dtype=int)
#         #idx_arr[np.where(idx_arr[:,0] < rc_min),0] = 0
#         #idx_arr[np.where(idx_arr[:,0] > r_max),0] = 0
#         #idx_arr[np.where(idx_arr[:,1] < rc_min),1] = 0
#         #idx_arr[np.where(idx_arr[:,1] > c_max),1] = 0

#         #idx_arr = idx_arr[np.where(np.any(idx_arr,-1))]

#         print(idx_arr.shape)

#         for r,c in idx_arr:
#             sum_arr[r-radius:r+radius,c-radius:c+radius] += map[r,c] * weight
#             count_arr[r-radius:r+radius,c-radius:c+radius] += [weight] * 3

#     count_arr[count_arr == 0] = 1   # Avoid dividing by 0

#     return sum_arr / count_arr