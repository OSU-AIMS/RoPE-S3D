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

am = Predictor(ds_factor=8)
ds = Dataset('set10')

start = 0
end = 1000

print("Copying Data...")

target_imgs = np.zeros((end-start,720,1280,3),np.uint8)
target_depths = np.zeros((end-start,720,1280))

og_imgs = np.copy(ds.og_img[start:end])
seg_img = np.copy(ds.seg_img[start:end])
dms = np.copy(ds.depthmaps[start:end])
cam_poses = np.copy(ds.camera_pose[start:end])
angles = np.copy(ds.angles[start:end])
angles[:,:3] += (np.random.rand(*(angles[:,:3].shape)) - .5) * 1

for i in range(end-start):
    target_imgs[i,] = seg_img[i]
    target_depths[i] = dms[i]

out = []

for idx in tqdm(range(end-start)):
    out.append(am.run(og_imgs[idx], target_imgs[idx], target_depths[idx], cam_poses[idx]))

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
