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
    Make Realsense Intrinsics from presets that are commonly used.

    Arguments:
    preset: string of the preset to use
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

    Arguments:
    resolution: Camera resolution (tuple)
    pp: pixels per meter (tuple)(x,y)
    f: focal plan position (tuple)(x,y)
    model: distortion model to use (rs.distortion)
    coeffs: distortion coeffs
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
    Python copy of the C++ realsense sdk function.

    Expects n x 3 array of points to project.
    https://github.com/IntelRealSense/librealsense/blob/master/include/librealsense2/rsutil.h

    Arguments:
    intrin: camera intrinsics (rs.intrinsics)
    points: n x 3 array of points to project in x,y,z format
    correct_distortion: bool, use distorsion correction of intrinsics
    """
    x = points[0] / points[2]
    y = points[1] / points[2]

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
    pixel[0] = x * intrin.fx + intrin.ppx
    pixel[1] = y * intrin.fy + intrin.ppy

    return pixel


def proj_point_to_pixel_map(intrin, points):
    """
    Creates an indexmap from a pointmap.

    Adaptation of proj_point_to_pixel().
    Cannot correct distortion.

    Arguments:
    intrin: intrinsics to use for projection
    points:  a x b x 3 array of points structured as x,y,z
    """
    x = points[...,0] / points[...,2]
    y = points[...,1] / points[...,2]

    x = np.nan_to_num(x)
    y = np.nan_to_num(y)

    pixel = np.zeros((*points.shape[0:2],2))
    pixel[...,0] = x * intrin.fx + intrin.ppx
    pixel[...,1] = y * intrin.fy + intrin.ppy

    pixel = pixel.astype(int)

    pixel[...,0] = np.clip(pixel[...,0],0,719)
    pixel[...,1] = np.clip(pixel[...,1],0,1279)

    return pixel




def deproj_pixel_to_point(intrin, pixels, depths):
    """
    Python copy of the C++ realsense sdk function.

    https://github.com/IntelRealSense/librealsense/blob/master/include/librealsense2/rsutil.h

    Arguments:
    intrin: camera intrinsics (rs.intrinsics)
    pixels: n x 2 array of points to project in x,y
    depths: array of depths
    """
    x = (pixels[0] - intrin.ppx) / intrin.fx
    y = (pixels[1] - intrin.ppy) / intrin.fy

    point = np.zeros((pixels.shape[0],3))

    point[0] = depths * x
    point[1] = depths * y
    point[2] = depths

    return point




def deproj_depthmap_to_pointmap(intrin, depthmap, depth_scale = 0, x_offset = 0, y_offset = 0):
    """
    BROKEN
    Deprojects an entire depthmap into a corresponding pointmap, assuming the same camera intrinsics

    Arguments:
    intrin: intrinsics to use to project
    depthmap: depthmap to project
    depth_scale: Multiplier to apply to depthmap if not already applied
    """

    depthmap = np.array(depthmap)

    point_map = np.zeros((*depthmap.shape,3))

    r_idx = np.arange(depthmap.shape[0])
    c_idx = np.arange(depthmap.shape[1])

    if depth_scale != 0:
        depthmap *= depth_scale
    depths = depthmap[r_idx,c_idx]

    ##################### Switch r and c?
    x = (c_idx + x_offset - intrin.ppx) / intrin.fx 
    y = (r_idx + y_offset - intrin.ppy) / intrin.fy 

    point_map[r_idx, c_idx, 0] = depths * x
    point_map[r_idx, c_idx, 1] = depths * y
    point_map[r_idx, c_idx, 2] = depths

    return point_map



def deproj_depthmap_to_pointmap_different(intrin_d, intrin_c, depthmap, depth_scale = 0):
    """
    Deprojects an entire depthmap into a corresponding pointmap with differing intrinsics

    Arguments:
    intrin_d: input depth intrinsics
    intirn_c: output color intrinsics
    depthmap: depthmap to project
    depth_scale: multipler to apply to depthmap if not already applied
    """

    depthmap = np.array(depthmap, dtype=np.float64)

    point_map = np.zeros((*depthmap.shape,3))

    r_idx = np.repeat(np.arange(depthmap.shape[0]),1280)
    c_idx = np.tile(np.arange(depthmap.shape[1]),720)

    if depth_scale != 0:
        depthmap *= depth_scale
    #depths = depthmap[r_idx,c_idx]
    depths = depthmap.flatten()

    ##################### Switch r and c?
    x = (c_idx - intrin_d.ppx) / intrin_d.fx 
    y = (r_idx - intrin_d.ppy) / intrin_d.fy 

    point_map[r_idx, c_idx, 0] = depths * x
    point_map[r_idx, c_idx, 1] = depths * y
    point_map[r_idx, c_idx, 2] = depths

    point_map[...,0] -= .0175   # Correct placement

    pixel_map = proj_point_to_pixel_map(intrin_c,point_map)
    cvt_pointmap = np.zeros((intrin_c.height, intrin_c.width,3))
    for r in range(pixel_map.shape[0]):
        for c in range(pixel_map.shape[1]):
            cvt_pointmap[pixel_map[r,c,0],pixel_map[r,c,1]] = point_map[r,c]
    
    return cvt_pointmap





# def generateMaps(points, intrin_type = '1280_720_color'):
#     intrin = makeIntrinsics(intrin_type)

#     # Instead of an RGB/BGR array, this is an XYZ array
#     sum_arr = np.zeros((intrin.height, intrin.width, 3))
#     count_arr = np.zeros((intrin.height, intrin.width, 3))

#     points_idx = np.array(np.round(proj_point_to_pixel(intrin, points)), dtype=int)

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
