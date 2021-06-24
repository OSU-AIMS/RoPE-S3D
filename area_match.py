# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from robotpose import Dataset, Predictor
import numpy as np
from tqdm import tqdm

ds = Dataset('set10')
am = Predictor(ds_factor=8, camera_pose=ds.camera_pose[0], preview=True, base_intrin = ds.attrs['color_intrinsics'], do_angles='SLU')


start = 180
end = 200

print("Copying Data...")
target_depths = np.copy(ds.depthmaps[start:end])
og_imgs = np.copy(ds.og_img[start:end])
cam_poses = np.copy(ds.camera_pose[start:end])

out = []

for idx in tqdm(range(end-start)):
    am.run(og_imgs[idx], target_depths[idx], cam_poses[idx])
