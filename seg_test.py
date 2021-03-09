import cv2
import os
from robotpose.segmentation import RobotSegmenter
a = RobotSegmenter()
for i in range(1,10):
    img, b = a.segmentImage(fr'C:\Users\exley\OneDrive\Documents\GitHub\DeepPoseRobot\data\raw\set6_slu\202103020005{i}_og.png',fr'C:\Users\exley\OneDrive\Documents\GitHub\DeepPoseRobot\data\raw\set6_slu\202103020005{i}_full.ply')
    cv2.imwrite(os.path.join("seg_test",f"{i}.png"),img)
