import cv2
import os
from robotpose.data.segmentation import RobotSegmenter
import robotpose.paths as p
import numpy as np
a = RobotSegmenter(intrinsics='1280_720_color', model_path=os.path.join('models/segmentation','C.h5'))
for i in range(1,10):
    mask, b = a.segmentImage(fr'data\raw\extracted\set6_slu\202103020008{i}_og.png')
    mask = mask.astype(np.uint8) * 255
    cv2.imshow('test',mask)
    cv2.waitKey(0)