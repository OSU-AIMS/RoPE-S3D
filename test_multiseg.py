# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import pixellib
import cv2
import numpy
from pixellib.instance import custom_segmentation

classes = ["BG","base_link","link_s", "link_l", "link_u","link_r","link_b"]

test_video = custom_segmentation()
test_video.inferConfig(num_classes=6, class_names=classes)
test_video.load_model("models/segmentation/multi/A.h5")
test_video.process_video("data/set10/og_vid.avi", show_bboxes = False,  output_video_name="output/multiseg_test.avi", frames_per_second=15)

# video = cv2.VideoCapture("data/set10/og_vid.avi")
# ret, frame = video.read()

# def reorganize_by_link(r, class_names):
#     out = {}
#     for idx in range(len(r['class_ids'])):
#         id = r['class_ids'][idx]
#         out[class_names[id]] = {
#             'roi':r['rois'][idx],
#             'confidence':r['scores'][idx],
#             'mask':r['masks'][idx]
#             }
#     return out

# r, output = test_video.segmentImage(frame,process_frame=True)
# print(reorganize_by_link(r,classes))