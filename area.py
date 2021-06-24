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

angs = 'SLU'

ds = Dataset('set10')
am = Predictor(ds_factor=4, camera_pose=ds.camera_pose[0], preview=False, base_intrin = ds.attrs['color_intrinsics'], do_angles=angs)

starting_points = True

div_size = 100

out = []

with tqdm(total=ds.length) as pbar:
    for start in range(0,1000,div_size):
        end = start+div_size

        target_depths = np.zeros((div_size,720,1280))

        og_imgs = np.copy(ds.og_img[start:end])
        dms = np.copy(ds.depthmaps[start:end])
        cam_poses = np.copy(ds.camera_pose[start:end])

        for idx in range(div_size):
            out.append(am.run(og_imgs[idx], dms[idx], cam_poses[idx]))
            pbar.update(1)

out = np.array(out)

g = Grapher(angs,out,np.copy(ds.angles))
g.plot()
g.plot(20)
g.plot(10)
