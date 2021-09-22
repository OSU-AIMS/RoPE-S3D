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

import numpy as np

from robotpose.training.planning import Planner

def run(ang, file, num, noise):

    if not file.endswith('.npy'):
        file += '.npy'

    p = Planner()

    if noise == 0:
        grid = p.basicGrid(ang,num)
    else:
        grid = p.noisyGrid(ang,num,noise)

    np.save(file,grid)
    logging.info(f"Saved to {file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-num', type=int, default=1000, help="Max of synthetic poses to create.")
    parser.add_argument('-file', type=str, default='plan', help="File to save poses to.")
    parser.add_argument('-angs', type=str, default='SLU',help="The joints to vary.")
    parser.add_argument('-noise', type=float, default=0 ,help="Noise to add in radians.")

    args = parser.parse_args()
    run(args.ang, args.file, args.num + 1, args.noise)
