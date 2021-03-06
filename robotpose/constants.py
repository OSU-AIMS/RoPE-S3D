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
MAX_LINKS = 7


PATH_JSON_PATH = r'data/paths.json'

JSON_LINK_FILE = r"\\marvin\ROPE\joint_states.json"

##################################### Crops
CROP_RENDER_WEIGHTING = [6,3,3,0,1,0]   # Higher numbers indicate more weight on that joint for rendering
CROP_VARYING = 'SLUB'   # Joints to vary for crop calculation
CROP_MAX_PER_JOINT = 50    # Max poses for a single joint
CROP_SEC_ALLOTTED_APPROX = 20   # Approx number of seconds allowed for each crop rendering stage calculation
CROP_PADDING = 10


##################################### Lookups

GPU_MEMORY_ALLOWED_FOR_LOOKUP = 0.1 # Depending on hardware, this my vary. ~10% seems to work, but anything ~25%+ will overallocate for calculations
LOOKUP_NAME_LENGTH = 5
LOOKUP_MAX_DIV_PER_LINK = 200
LOOKUP_JOINTS = 'SLU'   # SL is also usable
LOOKUP_NUM_RENDERED = 6 # 3 or 4 for SL


##################################### Segmentation Models

MODELDATA_FILE_NAME = 'ModelData.json'
NUM_MODELS_TO_KEEP = 3  # If a model has more than this number of stored checkpoints, they will be deleted.
MODEL_NAME_LENGTH = 4


##################################### Wizard Settings

WIZARD_DATASET_PREVIEW = True   # Set to false to reduce lag caused by dataset previewing


##################################### Verifier

VERIFIER_ALPHA = .7 # Weight to place on images in verifier
VERIFIER_SELECTED_GAMMA = -50   # Amount to add to R/G/B Channels of a selected image. Usually negative.
VERIFIER_SCALER = 1.5 # Scale factor of thumbnails. Overall scale is this divided by THUMBNAIL_DS_FACTOR
VERIFIER_ROWS = 4   # Rows of images present in Verifier
VERIFIER_COLUMNS = 4    # Columns of images present in Verifier


##################################### Datasets

VIDEO_FPS = 15  # Default video frames per second
THUMBNAIL_DS_FACTOR = 6 # Factor to downscale images by for thumbnails. Larger numbers yield smaller images
DEFAULT_CAMERA_POSE = [0, -1.5, .75, 0, 0, 0]   # Base camera pose to fill new datasets with before alignment

##################################### Rendering


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