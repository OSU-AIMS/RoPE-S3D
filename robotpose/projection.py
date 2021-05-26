# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import numpy as np
import pyrealsense2 as rs
import pyrender
import re

FLT_EPSILON = 1

class Intrinsics():

    def __init__(self, input: str = None):
        """Create intrinsics object

        Parameters
        ----------
        input : str, optional
            Preset intrinsics or string representation of intrinsics to replicate, by default None
        """
        self.bases = ['1280_720_color', '1280_720_depth','640_480_color','640_480_depth']

        if input is not None:
            is_preset = False

            for base in self.bases:
                if input == base or (base + '_') in input:
                    is_preset = True
                    break

            if is_preset:
                self.fromPreset(input)
            else:
                self.fromString(input)


    def fromString(self, input: str):
        """Configure intrinsics from the string representation of said intrinsics.

        Parameters
        ----------
        input : str
            String representation of desired intrinsics
        """

        integer = r'[1-9][0-9]*'
        decimal = r'[0-9]*(\.[0-9]*)?'

        resolution_re = rf'({integer}) *x *({integer})'
        pp_re = rf'p\[(?P<x> *{decimal})(?P<y> +{decimal})\]'
        f_re = rf'f\[(?P<x> *{decimal})(?P<y> +{decimal})\]'
        model_re = r'\] +(?P<model>[a-z ]*) +\['
        coeff_re = rf'\[(?P<a> *{decimal} +)(?P<b>{decimal} +)(?P<c>{decimal} +)(?P<d>{decimal} +)(?P<e>{decimal} *)\]'

        self.resolution = tuple([int(x) for x in re.search(resolution_re, input).groups()])
        self.pp = tuple([float(x) for x in re.search(pp_re, input).groupdict().values()])
        self.f = tuple([float(x) for x in re.search(f_re, input).groupdict().values()])

        model_name = [x for x in re.search(model_re, input, re.IGNORECASE).groupdict().values()][0]
        self.model = {'Brown Conrady': rs.distortion.brown_conrady,
        'Inverse Brown Conrady': rs.distortion.inverse_brown_conrady,
        'Ftheta': rs.distortion.ftheta,
        'Kannala Brandt4': rs.distortion.kannala_brandt4,
        'Modified Brown Conrady': rs.distortion.modified_brown_conrady,
        'None': rs.distortion.none,
        }.get(model_name)
        
        self.coeffs = [float(x) for x in re.search(coeff_re, input).groupdict().values()]


    def fromPreset(self, preset: str = '1280_720_color'):
        """Configure to commonly-used intrinsics presets.

        Parameters
        ----------
        preset : str, optional
            String of the preset to use, by default '1280_720_color'
        
        Presets can be downscaled by adding '_x' to the end with x being a downscale factor
        """

        def get_details(preset):
            if preset == '1280_720_color':
                res = (1280,720)
                pp = (638.391,361.493)
                f = (905.23, 904.858)
            elif preset == '1280_720_depth':
                res = (1280,720)
                pp = (639.459,359.856)
                f = (635.956, 635.956)
            elif preset == '640_480_color':
                res = (640,480)
                pp = (320.503,237.288)
                f = (611.528,611.528)
            elif preset == '640_480_depth':
                res = (640,480)
                pp = (321.635,241.618)
                f = (385.134,385.134)
            return res, pp, f

        self.model = rs.distortion.brown_conrady
        self.coeffs = [0,0,0,0,0]

        for base in self.bases:
            if preset == base:
                self.resolution, self.pp, self.f = get_details(preset)
                return

            elif (base + '_') in preset:
                ds_factor = int(preset.replace((base + '_'),''))
                self.resolution, self.pp, self.f = get_details(base)
                downscale_res = [x/ds_factor for x in self.resolution]
                res_is_valid = [int(x) == round(x) for x in downscale_res]
                if sum(res_is_valid) != 2:
                    raise ValueError(f"Downscaling by a factor of {ds_factor} is not valid for this resolution.")
                else:
                    self.resolution = tuple([x//ds_factor for x in self.resolution])
                    self.pp = tuple([x/ds_factor for x in self.pp])
                    self.f = tuple([x/ds_factor for x in self.f])
                    return


        raise ValueError(f"Preset must be one of: {self.bases}\nDownscaling a preset can be done by appending '_x' to a preset where x is a valid number.")


    @property
    def rs(self):
        """Intrinsics in pyrealsense2 form.

        Returns
        -------
        pyrealsense2.pyrealsense2.intrinsics
            Intrinsics compatible with pyrealsense2
        """

        a = rs.intrinsics()
        a.width = max(self.resolution)
        a.height = min(self.resolution)
        a.ppx = self.pp[0]
        a.ppy = self.pp[1]
        a.fx = self.f[0]
        a.fy = self.f[1]
        a.coeffs = self.coeffs
        a.model = self.model
        return a

    @property
    def pyrender_camera(self):
        """Pyrender camera from intrinsics

        Returns
        -------
        pyrender.IntrinsicsCamera
            Camera from specifified intrinsics
        """
        return pyrender.IntrinsicsCamera(cx=self.pp[0], cy=self.pp[1], fx=self.f[0], fy=self.f[1])
    
    @property
    def width(self) -> int:
        return max(self.resolution)

    @property
    def height(self) -> int:
        return min(self.resolution)

    def __str__(self) -> str:
        return str(self.rs)

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)






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
    points[...,2] = points[...,2][points[...,2] == 0] = 1   # Avoid dividing by 0
    x = points[...,0] / points[...,2]
    y = points[...,1] / points[...,2]

    x = np.nan_to_num(x).astype(np.float64)
    y = np.nan_to_num(y).astype(np.float64)


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


def deproj_depthmap_to_pointmap(intrin, depthmap, depth_scale = 0):
    """
    Deprojects an entire depthmap into a corresponding pointmap, assuming the same camera intrinsics

    Arguments:
    intrin: intrinsics to use to project
    depthmap: depthmap to project
    depth_scale: Multiplier to apply to depthmap if not already applied
    """

    depthmap = np.array(depthmap, dtype=np.float64)

    point_map = np.zeros((*depthmap.shape,3))

    r_idx = np.repeat(np.arange(depthmap.shape[0]),1280)
    c_idx = np.tile(np.arange(depthmap.shape[1]),720)

    if depth_scale != 0:
        depthmap *= depth_scale
    depths = depthmap.flatten()

    # Something is a bit messed up here, have to investigate
    x = (c_idx - intrin.ppx) / intrin.fx 
    y = (r_idx - intrin.ppy) / intrin.fy 

    point_map[r_idx, c_idx, 0] = depths * x
    point_map[r_idx, c_idx, 1] = depths * -y
    point_map[r_idx, c_idx, 2] = depths

    return point_map


# def deproj_depthmap_to_pointmap_different(intrin_d, intrin_c, depthmap, depth_scale = 0):
#     """
#     Deprojects an entire depthmap into a corresponding pointmap with differing intrinsics

#     Arguments:
#     intrin_d: input depth intrinsics
#     intirn_c: output color intrinsics
#     depthmap: depthmap to project
#     depth_scale: multipler to apply to depthmap if not already applied
#     """

#     depthmap = np.array(depthmap, dtype=np.float64)

#     point_map = np.zeros((*depthmap.shape,3))

#     r_idx = np.repeat(np.arange(depthmap.shape[0]),1280)
#     c_idx = np.tile(np.arange(depthmap.shape[1]),720)

#     if depth_scale != 0:
#         depthmap *= depth_scale
#     depths = depthmap.flatten()

#     x = (r_idx - intrin_d.ppx) / intrin_d.fx 
#     y = (c_idx - intrin_d.ppy) / intrin_d.fy 

#     point_map[r_idx, c_idx, 0] = depths * x
#     point_map[r_idx, c_idx, 1] = depths * y
#     point_map[r_idx, c_idx, 2] = depths

#     point_map[...,0] -= .0175   # Correct placement

#     pixel_map = proj_point_to_pixel_map(intrin_c,point_map)
#     cvt_pointmap = np.zeros((intrin_c.height, intrin_c.width,3))
#     for r in range(pixel_map.shape[0]):
#         for c in range(pixel_map.shape[1]):
#             cvt_pointmap[pixel_map[r,c,0],pixel_map[r,c,1]] = point_map[r,c]
    
#     return cvt_pointmap




def fill_hole(arr, r, c, rad):
    """
    expects n x n x 3 array
    """

    rc_dist = np.zeros((arr.shape[0], arr.shape[1],2))
    rc_dist[...,0] = np.hstack([np.arange(r,r-rc_dist.shape[0], -1).reshape(rc_dist.shape[0],1)] * rc_dist.shape[1])
    rc_dist[...,1] = np.vstack([np.arange(c,c-rc_dist.shape[1], -1)] * rc_dist.shape[0])

    rc_dist[r,c] = 100

    weight = np.power((np.square(rc_dist[...,0]) + np.square(rc_dist[...,1])), -1.5)
    
    include = np.zeros((arr.shape[0], arr.shape[1]),bool)
    include[r-rad:r+rad,c-rad:c+rad] = True
    
    is_val = np.any(arr, -1)
    is_val *= include

    pred = np.zeros(3)

    for idx in range(3):
        r_grad, c_grad = np.gradient(arr[...,idx])
        preds = arr[...,idx] + rc_dist[...,0]*r_grad + rc_dist[...,1]*c_grad

        preds_weighted = preds*weight

        weight_sum = np.sum(weight[np.where(is_val)])
        pred_sum = np.sum(preds_weighted[np.where(is_val)])

        pred[idx] = pred_sum / weight_sum
        
    return pred

