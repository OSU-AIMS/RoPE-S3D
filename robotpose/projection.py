# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import re
from typing import Union

import numpy as np
import pyrealsense2 as rs
import pyrender


class Intrinsics():

    def __init__(self, input: Union[str, 'Intrinsics'] = None):
        """Create intrinsics object

        Parameters
        ----------
        input : Union[str, robotpose.Intrinsics], optional
            Preset intrinsics or string representation of intrinsics to replicate, by default None, by default None
        """

        self.bases = ['1280_720_color', '1280_720_depth','640_480_color','640_480_depth']

        if input is not None:
            input = str(input)
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
                self.downscale(ds_factor)
                return

        raise ValueError(f"Input {preset} not valid.\nPreset must be one of: {self.bases}\nDownscaling a preset can be done by appending '_x' to a preset where x is a valid number.")


    def downscale(self, ds_factor):
        assert ds_factor >= 1,"Downscaling by a factor of less than 1 (upscaling) is not supported."
        downscale_res = [x/ds_factor for x in self.resolution]
        res_is_valid = [int(x) == round(x) for x in downscale_res]
        if sum(res_is_valid) != 2:
            raise ValueError(f"Downscaling by a factor of {ds_factor} is not valid for this resolution. This yields {downscale_res} as a resolution, which cannot be interpreted.")
        else:
            self.resolution = tuple([x//ds_factor for x in self.resolution])
            self.pp = tuple([x/ds_factor for x in self.pp])
            self.f = tuple([x/ds_factor for x in self.f])


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

    @property
    def size(self) -> int:
        return np.prod(np.array(self.resolution))

    def __str__(self) -> str:
        return str(self.rs)

    def __eq__(self, other) -> bool:
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)
