# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley


import argparse
import logging
import os
import re

import numpy as np

from robotpose import Dataset, Grapher
from robotpose.prediction.analysis import JointDistance
from robotpose.utils import str_to_arr


# Disable OpenGL and Tensorflow info messages (get annoying)
logging.getLogger("OpenGL.arrays.arraydatatype").setLevel(logging.WARNING)
logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf


def run(args):
    file = args.file

    if not file.endswith('.npy'):
        file+='.npy'

    results = np.load(file)

    if results.shape[0] == 2:
        angles = results[0]
        preds = results[1]
    else:
        name = re.search(r'_set.+_',file)
        if name is not None:
            dataset = name.group(0)[1:][:-1]
        else:
            name = re.search(r'_set.+\.npy',file)
            dataset = name.group(0)[1:][:-4]

        ds = Dataset(dataset)
        preds = results
        angles = np.copy(ds.angles)

    idx_to_sort = np.where(str_to_arr(args.sort_by))[0][0]
    idx_to_sort = 0
    indicies = np.argsort(angles[...,idx_to_sort])


    g = Grapher(args.angs,preds[indicies],angles[indicies])
    g.plot(20)

    j = JointDistance()
    j.plot(preds[indicies],angles[indicies],.25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help="The file to view.")
    parser.add_argument('-sort_by', type=str, default='S', help="Joint to sort by.")
    parser.add_argument('-angs', type=str, default='SLU',help="The joints to predict.")

    args = parser.parse_args()
    run(args)

