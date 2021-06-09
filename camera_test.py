# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from robotpose import Dataset
from robotpose.prediction.camera import CameraPredictor
import numpy as np


pred = CameraPredictor([0,-1.5,1,0,0,0],ds_factor=8, preview=False)#, save_to='output/projection_viz.avi')
ds = Dataset('set10')

idxs = [50,167,583,901,224]
# idxs = [50,0,25,75,99]
#idxs = [x for x in range(0,100,5)]

print("Copying Data...")
idxs_sorted = idxs.copy()
idxs_sorted.sort()
idx_map = [idxs.index(x) for x in idxs_sorted]

target_depth = np.copy(ds.depthmaps[idxs_sorted])
og_img = np.copy(ds.og_img[idxs_sorted])
angles = np.copy(ds.angles[idxs_sorted])

target_depth = target_depth[idx_map]
og_img = og_img[idx_map]
angles = angles[idx_map]

predicted = pred.run(og_img, target_depth, angles)
predicted = pred.run(og_img, target_depth, angles, predicted)

print(f"Actual: {ds.camera_pose[idxs[0]]}\nPredicted: {predicted}")