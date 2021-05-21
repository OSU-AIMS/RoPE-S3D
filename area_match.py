# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from robotpose import Dataset, Predictor
from robotpose.prediction.predict import TimePredictor
import numpy as np
from tqdm import tqdm

am = TimePredictor(ds_factor=8, preview=False)#, save_to='output/projection_viz.avi')
ds = Dataset('set10')

start = 98
end = 103

print("Copying Data...")
target_imgs = np.copy(ds.seg_img[start:end])
target_depths = np.copy(ds.depthmaps[start:end])
og_imgs = np.copy(ds.og_img[start:end])
cam_poses = np.copy(ds.camera_pose[start:end])

out = []

for idx in tqdm(range(end-start)):
    am.run(og_imgs[idx], target_imgs[idx], target_depths[idx], cam_poses[idx])
