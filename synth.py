# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from robotpose import Dataset, SyntheticPredictor

angs = 'SLU'
dataset = 'set63'
number_of_poses = 10000


ds = Dataset(dataset)
synth = SyntheticPredictor(ds.camera_pose[0],'1280_720_color',8,angs)
synth.run_batch(number_of_poses)
