# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import numpy as np
from tqdm import tqdm

from robotpose import Dataset, Predictor
from robotpose.utils import Grapher

ds = Dataset('set10')
am = Predictor(ds_factor=8, preview=False, base_intrin = ds.attrs['color_intrinsics'])

starting_points = True

div_size = 100

out = []

for start in range(0,1000,div_size):
    end = start+div_size

    print("Copying Data...")

    target_depths = np.zeros((div_size,720,1280))

    og_imgs = np.copy(ds.og_img[start:end])
    dms = np.copy(ds.depthmaps[start:end])
    cam_poses = np.copy(ds.camera_pose[start:end])

    for idx in tqdm(range(div_size)):
        out.append(am.run(og_imgs[idx], dms[idx], cam_poses[idx]))

out = np.array(out)

g = Grapher('SLU',out,np.copy(ds.angles))
g.plot()
g.plot(20)
g.plot(10)
