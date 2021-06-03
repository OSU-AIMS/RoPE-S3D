# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import pixellib
from pixellib.instance import custom_segmentation
from robotpose.training import ModelManager

classes = ["BG","base_link","link_s", "link_l", "link_u","link_r","link_b"]

mm = ModelManager()
print("\n\n",mm.dynamicLoad('link', train_ratio = 0))

test_video = custom_segmentation()
test_video.inferConfig(num_classes=6, class_names=classes)
test_video.load_model(mm.dynamicLoad('link', train_ratio = 0))
test_video.process_video("data/set10/og_vid.avi", show_bboxes = False,  output_video_name="output/multiseg_test.avi", frames_per_second=15)

