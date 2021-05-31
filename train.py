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
from robotpose.training import ModelManager
from robotpose import Paths as p

# Eager Exec. is enabled when importing robotpose; disable
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


def train(dataset, mode, batch):
    ds = Dataset(dataset)

    # Get names of classes for modeldata
    class_names = [x for x in DatasetRenderer(dataset, mode = {'body':'seg_full','link':'seg'}.get(mode)).color_dict]

    # Create a new model location
    mm = ModelManager()
    dest = mm.allocateNew(mode, dataset, class_names)

    # Configure training
    train_maskrcnn = instance_custom_training()
    train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes = len(class_names), batch_size = batch)
    train_maskrcnn.load_pretrained_model(p().BASE_MODEL)

    # Train
    train_maskrcnn.load_dataset({'body':ds.body_anno_path,'link':ds.link_anno_path}.get(mode))
    train_maskrcnn.train_model(num_epochs = 300, augmentation=True,  path_trained_models = os.path.abspath(dest))

    mm.update()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default="set6", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('mode', type=str, default="link", choices=['link','body'], help="The type of model to train.")
    parser.add_argument('-batch',type=int, choices=[1,2,4,8,12,16], default=2, help="Batch size for training")
    args = parser.parse_args()

    train(args.dataset, args.mode, args.batch)