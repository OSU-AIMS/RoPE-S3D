# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley


import os
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from tqdm import tqdm
import open3d as o3d


FLT_EPSILON = 1



def makeIntrinsics(preset = '1280_720_color'):
    """
    Make Realsense Intrinsics from presets
    """

    valid = ['1280_720_color', '1280_720_depth','640_480_color','640_480_depth']
    if preset not in valid:
        raise ValueError(f"Res must be one of: {valid}")

    if preset == '1280_720_color':
        return intrin((1280,720), (638.391,361.493), (905.23, 904.858), rs.distortion.inverse_brown_conrady, [0,0,0,0,0])
    elif preset == '1280_720_depth':
        return intrin((1280,720), (639.459,359.856), (635.956, 635.956),rs.distortion.brown_conrady, [0,0,0,0,0])
    elif preset == '640_480_color':
        return intrin((640,480), (320.503,237.288), (611.528,611.528),rs.distortion.brown_conrady, [0,0,0,0,0])
    elif preset == '640_480_depth':
        return intrin((640,480), (321.635,241.618), (385.134,385.134),rs.distortion.brown_conrady, [0,0,0,0,0])




def intrin(resolution, pp, f, model, coeffs):
    """
    Makes psuedo-intrinsics for the realsense camera used.
    """
    a = rs.intrinsics()
    a.width = max(resolution)
    a.height = min(resolution)
    a.ppx = pp[0]
    a.ppy = pp[1]
    a.fx = f[0]
    a.fy = f[1]
    a.coeffs = coeffs
    a.model = model
    return a





def proj_point_to_pixel(intrin, points, correct_distortion = False):
    """
    Python copy of the C++ realsense sdk function
    https://github.com/IntelRealSense/librealsense/blob/master/include/librealsense2/rsutil.h
    Can take arrays as inputs to speed up calculations
    Expects n x 3 array of points to project
    """
    x = points[:,0] / points[:,2]
    y = points[:,1] / points[:,2]

    if correct_distortion:
        if intrin.model == rs.distortion.inverse_brown_conrady or intrin.model == rs.distortion.modified_brown_conrady:

            r_two = np.square(x) + np.square(y)

            f = 1 + intrin.coeffs[0] * r_two + intrin.coeffs[0] * np.square(r_two) + intrin.coeffs[4] * np.power(r_two,3)

            x *= f
            y *= f

            dx = x + 2*intrin.coeffs[2]*x*y + intrin.coeffs[3]*(r_two + 2*np.square(x))
            dy = y + 2*intrin.coeffs[3]*x*y + intrin.coeffs[2]*(r_two + 2*np.square(y))

            x = dx
            y = dy

        elif intrin.model == rs.distortion.brown_conrady:

            r_two = np.square(x) + np.square(y)

            f = 1 + intrin.coeffs[0] * r_two + intrin.coeffs[1] * np.square(r_two) + intrin.coeffs[4] * np.power(r_two,3)

            xf = x*f
            yf = y*f

            dx = xf + 2 * intrin.coeffs[2] * x*y + intrin.coeffs[3] * (r_two + 2 * np.square(x))
            dy = yf + 2 * intrin.coeffs[3] * x*y + intrin.coeffs[2] * (r_two + 2 * np.square(y))

            x = dx
            y = dy

        elif intrin.model == rs.distortion.ftheta:
            r = np.sqrt(np.square(x) + np.square(y))

            if r < FLT_EPSILON:
                r = FLT_EPSILON

            rd = (1.0 / intrin.coeffs[0] * np.arctan(2 * r* np.tan(intrin.coeffs[0] / 2.0)))

            x *= rd / r
            y *= rd / r

        elif intrin.model == rs.distortion.kannala_brandt4:

            r = np.sqrt(np.square(x) + np.square(y))

            if (r < FLT_EPSILON):
                r = FLT_EPSILON

            theta = np.arctan(r)

            theta_two = np.square(theta)

            series = 1 + theta_two*(intrin.coeffs[0] + theta_two*(intrin.coeffs[1] + theta_two*(intrin.coeffs[2] + theta_two*intrin.coeffs[3])))

            rd = theta*series
            x *= rd / r
            y *= rd / r

    pixel = np.zeros((points.shape[0],2))
    pixel[:,0] = x * intrin.fx + intrin.ppx
    pixel[:,1] = y * intrin.fy + intrin.ppy

    return pixel






def generateMaps(points, intrin_type = '1280_720_color'):
    intrin = makeIntrinsics(intrin_type)

    # Instead of an RGB/BGR array, this is an XYZ array
    sum_arr = np.zeros((intrin.height, intrin.width, 3))
    count_arr = np.zeros((intrin.height, intrin.width, 3))

    points_idx = np.array(np.round(proj_point_to_pixel(intrin, points)), dtype=int)

    points[:,2] *= -1

    points_idx[:,0] = np.round(np.clip(points_idx[:,0],0,intrin.width-1))
    points_idx[:,1] = np.round(np.clip(points_idx[:,1],0,intrin.height-1))

    for pixel, loc in zip(points_idx,points):
        px, py = pixel
        sum_arr[py,px] += loc
        count_arr[py,px] += [1]*3

    arr = sum_arr / count_arr
    arr = np.nan_to_num(arr,nan=0,posinf=0,neginf=0)

    return arr
