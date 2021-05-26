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
import random
import requests
import shutil

#import pixellib # If you don't include this, it won't work. No idea why.

from pixellib.custom_train import instance_custom_training

from robotpose import Dataset, DatasetRenderer
from robotpose.training import ModelManager
from robotpose.paths import Paths as p


def train(dataset, mode, batch):
    ds = Dataset(dataset)

    # print("Splitting up data...")
    # # Split set into validation and train
    # for folder in ['train', 'test']:
    #     path = os.path.join(ds.body_anno_path, folder)
    #     if os.path.isdir(path):
    #         shutil.rmtree(path)
    #     os.mkdir(path)

    # jsons = [x for x in os.listdir(ds.body_anno_path) if x.endswith('.json') and 'test' not in x and 'train' not in x]
    # random.shuffle(jsons)

    # valid_size = int(len(jsons) * valid)
    # valid_list = jsons[:valid_size]
    # train_list = jsons[valid_size:]

    # for lst, folder in zip([valid_list, train_list],['test','train']):
    #     path = os.path.join(ds.body_anno_path, folder)
    #     for file in lst:
    #         shutil.copy2(os.path.join(ds.body_anno_path, file), os.path.join(path, file))
    #         shutil.copy2(os.path.join(ds.body_anno_path, file.replace('.json','.png')), os.path.join(path, file.replace('.json','.png')))

    # print("Data Split.")
    # default_model_path = r'models/segmentation/mask_rcnn_coco.h5'



    # Get names of classes for modeldata
    class_names = [x for x in DatasetRenderer(dataset, mode = {'body':'seg_full','link':'seg'}.get(mode)).color_dict]

    # Create a new model location
    mm = ModelManager()
    dest = mm.allocateNew(mode, dataset, class_names)
    print(type(len(class_names)))

    # Configure training
    train_maskrcnn = instance_custom_training()
    train_maskrcnn.modelConfig(network_backbone = "resnet101", num_classes = len(class_names), batch_size = batch)
    train_maskrcnn.load_pretrained_model(p().BASE_MODEL)

    # Train
    train_maskrcnn.load_dataset({'body':ds.body_anno_path,'link':ds.link_anno_path}.get(mode))
    train_maskrcnn.train_model(num_epochs = 300, augmentation=True,  path_trained_models = os.path.abspath(dest))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default="set6", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('mode', type=str, default="link", choices=['link','body'], help="The type of model to train.")
    parser.add_argument('-batch',type=int, choices=[1,2,4,8,12,16], default=2, help="Batch size for training")
    args = parser.parse_args()

    train(args.dataset, args.mode, args.batch)