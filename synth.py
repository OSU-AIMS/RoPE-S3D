# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import argparse

from robotpose import Dataset, SyntheticPredictor

from robotpose.training.planning import Planner


def run(args):
    p = Planner()


    ds = Dataset(args.dataset)
    synth = SyntheticPredictor(ds.camera_pose[0],args.intrinsics,args.ds_factor,args.angs,noise=args.noise)
    synth.run_batch(args.num,args.file)

    # synth.run_batch_poses(p.basicGrid('SLU',1001), args.file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="The dataset pose to use. Can be a partial name.")
    parser.add_argument('-num', type=int, default=2500, help="Number of synthetic poses to predict.")
    parser.add_argument('-file', type=str, default='synth_test', help="File to save results to.")
    parser.add_argument('-noise', action="store_true", help="Adds semi-realistic noise to depth images.")
    parser.add_argument('-ds_factor', type=int, default=8, choices=[1,2,4,6,8,10,12], help="Downsampling factor.")
    parser.add_argument('-angs', type=str, default='SLU',help="The joints to predict.")
    parser.add_argument('-intrinsics', type=str, default='1280_720_color',help="Base camera instrinsics to use.")

    args = parser.parse_args()
    run(args)
