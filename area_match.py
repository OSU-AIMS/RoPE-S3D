# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from robotpose import Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
from robotpose.experimental.area import ProjectionMatcher, ProjectionMatcherLookup


WIDTH = 800

am = ProjectionMatcherLookup(ds_factor=8, preview=True)
ds = Dataset('set10')

start = 140
end = 160

print("Copying Data...")
roi_start = np.copy(ds.rois[start:end,1])
target_imgs = np.zeros((end-start,720,1280,3),np.uint8)
target_depths = np.zeros((end-start,720,1280), np.float32)

og_imgs = np.copy(ds.og_img[start:end])
seg_img = np.copy(ds.seg_img[start:end])
dms = np.copy(ds.pointmaps[start:end,...,2])
cam_poses = np.copy(ds.camera_pose[start:end])

for i,s in zip(range(end-start),roi_start):
    target_imgs[i,:,s:s+WIDTH] = seg_img[i]
    target_depths[i,:,s:s+WIDTH] = dms[i]


out = []

for idx in range(end-start):
    cv2.imshow('Target',target_imgs[idx])
    cv2.waitKey(1)
    am.run(og_imgs[idx], target_imgs[idx], target_depths[idx], cam_poses[idx])



