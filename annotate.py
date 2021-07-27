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
import logging

# Disable OpenGL and Tensorflow info messages (get annoying with multiprocessing)
logging.getLogger("OpenGL.arrays.arraydatatype").setLevel(logging.WARNING)
logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf

from robotpose import AutomaticAnnotator, DatasetRenderer

def label(args):
    rend = DatasetRenderer(args.dataset)
    
    seg = AutomaticAnnotator(args.dataset, rend, not args.no_preview)
    seg.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default="set10", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('-no_preview', action="store_true", help="Disables preview.")
    args = parser.parse_args()
    label(args)