# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from robotpose.utils import Grapher
from robotpose import Dataset, Predictor
import numpy as np
from tqdm import tqdm

am = Predictor(ds_factor=8, preview=False)
ds = Dataset('set10')

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

out_dicts = []
for pred in out:
    out_dict = {}
    for i,key in zip(range(3),['S','L','U']):
        out_dict[key] = {}
        out_dict[key]['val'] = pred[i]
        out_dict[key]['percent_est'] = 0
    out_dicts.append(out_dict)

g = Grapher(['S','L','U'],out_dicts,np.copy(ds.angles))
g.plot()
g.plot(20)
