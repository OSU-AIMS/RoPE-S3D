import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import glob

from deepposekit.io import TrainingGenerator, DataGenerator

from deepposekit.models import DeepLabCut, StackedDenseNet, StackedHourglass, LEAP

from deepposekit.models import load_model

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from deepposekit.callbacks import Logger, ModelCheckpoint

from robotpose import paths as p


data_generator = DataGenerator(p.ds_mult)

train_generator = TrainingGenerator(generator=data_generator,
                                    downsample_factor=0,
                                    augmenter=None,
                                    sigma=5,
                                    validation_split=0.15, 
                                    use_graph=True,
                                    random_seed=1,
                                    graph_scale=1)

#model = StackedDenseNet(train_generator, n_stacks=2, growth_rate=32, pretrained=True)

#model = DeepLabCut(train_generator, backbone="resnet50")
#model = DeepLabCut(train_generator, backbone="mobilenetv2", alpha=0.35) # Increase alpha to improve accuracy
#model = DeepLabCut(train_generator, backbone="densenet121")

model = LEAP(train_generator)
#model = StackedHourglass(train_generator)


"""
Doesn't want to work on my machine for some reason
"""
# logger = Logger(validation_batch_size=2,
#     # filepath saves the logger data to a .h5 file
#     filepath=p.log
# )
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, verbose=1, patience=20)

model_checkpoint = ModelCheckpoint(
    p.model_mult,
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

callbacks = [early_stop, reduce_lr, model_checkpoint]

model.fit(
    batch_size=2,
    validation_batch_size=2,
    callbacks=callbacks,
    epochs=1000,
    n_workers=2,
    steps_per_epoch=None,
)