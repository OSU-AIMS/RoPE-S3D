# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley
import numpy as np
import logging as log


##################################### Verifier

VERIFIER_ALPHA = .7
VERIFIER_SELECTED_GAMMA = -50
VERIFIER_SCALER = 2
VERIFIER_ROWS = 5
VERIFIER_COLUMNS = 7


##################################### Datasets

VIDEO_FPS = 15
THUMBNAIL_DS_FACTOR = 6
DEFAULT_CAMERA_POSE = [0, -1.5, .75, 0, 0, 0]

###################################### Rendering


def default_render_color_maker(num:int):
    """Creates unique colors for rendering.

    Parameters
    ----------
    num : int
        Number of colors to generate. Should be larger than the number of meshes expected to use.
        For 6-axis robots, the minimum recommended number is 7.

    Returns
    -------
    List[List]
        num pairs of RGB triplets
    """
    if num < 7:
        log.warn('Fewer than 7 rendering colors are being generated. This may cause issues if a URDF with a 6+ axis robot is loaded.')

    b = np.linspace(0,255,num).astype(int) # Blue values are always unique

    g = [0] * b.size
    r = np.abs(255 - 2*b)

    colors = []
    for idx in range(num):
        colors.append([b[idx],g[idx],r[idx]])
    return colors

DEFAULT_RENDER_COLORS = default_render_color_maker(7)   # Increase if expecting to use more meshes/end effector