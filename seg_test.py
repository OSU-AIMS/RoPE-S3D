import cv2
import os
from robotpose.segmentation import RobotSegmenter
a = RobotSegmenter()
for i in range(1,10):
    img = a.segment(fr'C:\Users\exley\OneDrive\Documents\GitHub\DeepPoseRobot\data\raw\set6_slu\202103020005{i}_og.png',r'C:\Users\exley\OneDrive\Documents\GitHub\DeepPoseRobot\data\raw\set6_slu\2021030200001_full.ply')
    cv2.imwrite(os.path.join("seg_test",f"{i}.png"),img)