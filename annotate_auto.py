# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf

from robotpose.autoAnnotate import AutomaticKeypointAnnotator, AutomaticSegmentationAnnotator
from robotpose.render import Renderer


def label(dataset, skeleton, preview):
    rend = Renderer(dataset, skeleton)
    key = AutomaticKeypointAnnotator(dataset, skeleton, renderer = rend, preview = preview)
    key.run()
    del key
    seg = AutomaticSegmentationAnnotator(dataset, skeleton, renderer = rend, preview = preview)
    seg.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default="set6", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('skeleton', type=str, default="B", help="The skeleton to use for annotation.")
    parser.add_argument('--no_preview', action="store_true", help="Disables preview.")
    args = parser.parse_args()
    label(args.dataset, args.skeleton, not args.no_preview)