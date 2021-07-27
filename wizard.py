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
import sys
import logging

# Disable OpenGL and Tensorflow info messages (get annoying with multiprocessing)
logging.getLogger("OpenGL.arrays.arraydatatype").setLevel(logging.WARNING)
logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf

from robotpose import Wizard, Dataset, DatasetInfo


if __name__ == "__main__":

    if len(sys.argv) > 1:
        ds_info = DatasetInfo()
        parser = argparse.ArgumentParser()
        parser.add_argument("dataset", type=str, default=None, choices=ds_info.unique_sets, help="Dataset to run wizard on")
        parser.add_argument("-recompile", action='store_true', help="Reprocessing dataset from data stored in dataset")
        parser.add_argument("-rebuild", action='store_true', help="Recreate dataset from the raw data")
        args = parser.parse_args()
        ds = Dataset(args.dataset, recompile=args.recompile, rebuild=args.rebuild, permissions='a')
    else:
        wiz = Wizard()
        wiz.run()