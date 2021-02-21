import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import glob

from deepposekit.io import TrainingGenerator, DataGenerator
from deepposekit.augment import FlipAxis
import imgaug.augmenters as iaa
import imgaug as ia

from deepposekit.models import DeepLabCut, StackedDenseNet, StackedHourglass, LEAP

from deepposekit.models import load_model

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from deepposekit.callbacks import Logger, ModelCheckpoint


import time
from os.path import expanduser


image_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\2d"
ds_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\mult_ds.h5"
skeleton_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\mult_skeleton.csv"
log_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\log.h5"
model_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\model_LEAP_mult.h5"


data_generator = DataGenerator(ds_path)


train_generator = TrainingGenerator(generator=data_generator,
                                    downsample_factor=0,
                                    augmenter=None,
                                    sigma=5,
                                    validation_split=0.1, 
                                    use_graph=True,
                                    random_seed=1,
                                    graph_scale=1)

#model = StackedDenseNet(train_generator, n_stacks=2, growth_rate=32, pretrained=True)

#model = DeepLabCut(train_generator, backbone="resnet50")
#model = DeepLabCut(train_generator, backbone="mobilenetv2", alpha=0.35) # Increase alpha to improve accuracy
#model = DeepLabCut(train_generator, backbone="densenet121")

model = LEAP(train_generator)
#model = StackedHourglass(train_generator)



logger = Logger(validation_batch_size=2,
    # filepath saves the logger data to a .h5 file
    filepath=log_path
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, verbose=1, patience=20)

model_checkpoint = ModelCheckpoint(
    model_path,
    monitor="val_loss",
    # monitor="loss" # use if validation_split=0
    verbose=1,
    save_best_only=True,
)

early_stop = EarlyStopping(
    monitor="val_loss",
    # monitor="loss" # use if validation_split=0
    min_delta=0.001,
    patience=100,
    verbose=1
)

callbacks = [early_stop, reduce_lr, model_checkpoint, logger]
callbacks = [early_stop, reduce_lr, model_checkpoint]


model.fit(
    batch_size=2,
    validation_batch_size=2,
    callbacks=callbacks,
    #epochs=1000, # Increase the number of epochs to train the model longer
    epochs=1000,
    n_workers=2,
    steps_per_epoch=None,
)