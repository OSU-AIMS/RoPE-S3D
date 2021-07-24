from robotpose.projection import Intrinsics
from weakref import WeakKeyDictionary
from matplotlib.pyplot import imshow
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.gen_math_ops import imag
import robotpose
import numpy as np
import cv2
from tqdm import tqdm
from robotpose.data.segmentation import RobotSegmenter
from robotpose.training import models
from robotpose.training.models import ModelManager
from pixellib.instance import custom_segmentation
from robotpose.urdf import URDFReader
from robotpose.turbo_colormap import color_array
import matplotlib.pyplot as plt
from robotpose.utils import Grapher
from robotpose.simulation.lookup import RobotLookupCreator
import h5py

DS_SCALE = 8
MAX_DEPTH = 7
TRAIN_SIZE = 800


ds = robotpose.Dataset('set10')
# dim = ds.og_img.shape

# inp_img = np.copy(ds.og_img)
# inp_depth = np.copy(ds.depthmaps)
# data = np.zeros((ds.length,90,160),float)

u_reader = URDFReader()
# classes = ["BG"]
# classes.extend(u_reader.mesh_names[:6])
# mm = ModelManager()
# seg = custom_segmentation()
# seg.inferConfig(num_classes=6, class_names=classes)
# seg.load_model(mm.dynamicLoad(dataset='set10'))

# for idx in tqdm(range(ds.length)):
#     r,output = seg.segmentFrame((inp_img[idx]))
#     out = np.sum(r['masks'],-1).astype(bool)
#     depth = (inp_depth[idx].astype(float) * out.astype(float))

#     data[idx] = cv2.resize(depth,(160, 90))

# del inp_img,inp_depth

# data = np.clip(data,0,MAX_DEPTH) / MAX_DEPTH
# for idx in range(ds.length):
#     cv2.imshow("",color_array(data[idx]))
#     cv2.waitKey(1)

# np.save("test_data.npy",data)






# """Create a lookup"""
# i = Intrinsics(ds.intrinsics)
# i.downscale(8)

# r = RobotLookupCreator(ds.camera_pose[0],i)
# r.load_config(6,'SLU',np.array([45,45,45,1,1,1]))
# r.run('CNN_Lookup.h5',False,False)



# with h5py.File('CNN_Lookup.h5','r') as f:
#     angs = np.copy(f['angles'])
#     depths = np.copy(f['depth'])[:,:,:,np.newaxis]

#     print(angs.shape,depths.shape)

# angs /= 2 * np.pi

batch_size = 256
epochs = 50

# model = keras.models.Sequential()
# model.add(layers.Conv2D(64,6,activation='relu',input_shape=(160,90,1)))
# model.add(layers.MaxPool2D(2))
# model.add(layers.Conv2D(64,3,activation='relu'))
# model.add(layers.MaxPool2D(2))
# model.add(layers.Conv2D(64,3,activation='relu'))
# model.add(layers.MaxPool2D(2))
# model.add(layers.Conv2D(64,3,activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(12,activation='relu'))
# model.add(layers.Dense(12,activation='relu'))
# model.add(layers.Dense(3,activation='linear'))

# print(model.summary())

# loss = keras.losses.MeanSquaredError()
# optim = keras.optimizers.Adam(lr=.001)
# metrics = ['mae','mse']

# model.compile(optim,loss,metrics)

# model.fit(depths,angs[:,:3],batch_size,epochs,2,validation_batch_size=batch_size,validation_split=0.05)
# model.save('thisisatest')







data = np.load("test_data.npy")[:,:,:,np.newaxis]
print(data.shape)

# labels = np.zeros((ds.length,3))
# for idx in range(3):
#     labels[:,idx] = (np.copy(ds.angles[:,idx]) - u_reader.joint_limits[idx,0]) /  (u_reader.joint_limits[idx,1] - u_reader.joint_limits[idx,0]))

model = keras.models.load_model('thisisatest')

out = model.predict(data,batch_size,2)
# for idx in range(3):
#     out[:,idx] = (out[:,idx] * (u_reader.joint_limits[idx,1] - u_reader.joint_limits[idx,0]) + u_reader.joint_limits[idx,0])


new = np.zeros((ds.length,6))
new[:,:3] = out * np.pi * 2

g = Grapher('SLU',new,np.copy(ds.angles))
g.plot()