# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

# Skimage gives annoying FutureWarnings from transform\_warps.py:830
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

import argparse
import os

from pixellib.custom_train import instance_custom_training

from robotpose import Dataset, DatasetRenderer
from robotpose import Paths as p
from robotpose.training import ModelManager
from robotpose.data.annotation import refresh_split

# Eager Exec. is enabled when importing robotpose; disable
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def train(dataset, batch, cont, cont_from):
    
    refresh_split(dataset)
    ds = Dataset(dataset)

    # Get names of classes for modeldata
    class_names = [x for x in DatasetRenderer(dataset, mode = 'seg').color_dict]

    mm = ModelManager()

    base_model_path = None
    if cont or cont_from is not None:
        base_model_path = mm.dynamicLoad(dataset=(cont_from if cont_from is not None else dataset))
    if base_model_path is None:
        base_model_path = p().BASE_MODEL

    dest = mm.allocateNew(dataset, class_names)

    # Configure training
    train_maskrcnn = instance_custom_training()
    train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes = len(class_names), batch_size = batch)
    train_maskrcnn.load_pretrained_model(base_model_path)

    # Train
    train_maskrcnn.load_dataset(ds.link_anno_path)
    train_maskrcnn.train_model(num_epochs = 300, augmentation=True,  path_trained_models = os.path.abspath(dest))

    mm.update()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('-batch_size',type=int, choices=[1,2,4,8,12,16], default=2, help="Batch size for training")
    parser.add_argument('-cont',action='store_true', help="Continue latest trained model.")
    parser.add_argument('-cont_from', type=str, default=None, help="Last model to build from.")

    args = parser.parse_args()

    train(args.dataset, args.batch_size, args.cont, args.cont_from)
