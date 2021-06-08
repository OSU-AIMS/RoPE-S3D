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


pred = CameraPredictor(ds_factor=8, preview=True)#, save_to='output/projection_viz.avi')
ds = Dataset('set10')

idx = 180

print("Copying Data...")
target_depth = np.copy(ds.depthmaps[idx])
og_img = np.copy(ds.og_img[idx])
angles = np.copy(ds.angles[idx])

out = []

predicted = pred.run(og_img, target_depth, angles)


print(f"Actual: {ds.camera_pose[idx]}\nPredicted: {predicted}")