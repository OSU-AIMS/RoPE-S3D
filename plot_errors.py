# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import logging
import os
import re

import numpy as np

from robotpose import Dataset, Grapher
from robotpose.prediction.analysis import JointDistance

# Disable OpenGL and Tensorflow info messages (get annoying)
logging.getLogger("OpenGL.arrays.arraydatatype").setLevel(logging.WARNING)
logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf

file = 'synth_test.npy'

results = np.load(file)

print(results.shape[0])

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

    print(dataset)
    ds = Dataset(dataset)

    preds = results
    angles = np.copy(ds.angles)



IDX_TO_USE = 0
PERCENTILE_TO_SHOW = 99

# Sort by angle
indicies = np.argsort(angles[...,IDX_TO_USE])

out = np.sort(indicies)

# Graph angle offsets
g = Grapher('SLU',preds[indicies],angles[indicies])
g.plot(20)

diff = np.abs(preds - angles)

# Graph joint offsets
j = JointDistance()
j.plot(preds[indicies],angles[indicies],.25)
