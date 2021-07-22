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

DS_SCALE = 6
MAX_DEPTH = 7
TRAIN_SIZE = 800


ds = robotpose.Dataset('set10')
dim = ds.og_img.shape

# inp_img = np.copy(ds.og_img)
# inp_depth = np.copy(ds.depthmaps)
# data = np.zeros((ds.length,120,120),float)

# side_pad = (dim[2]-dim[1])//2

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
#     depth = (inp_depth[idx].astype(float) * out.astype(float))[:,side_pad:-side_pad]

#     data[idx] = cv2.resize(depth,(dim[1] // DS_SCALE, dim[1] // DS_SCALE))

# del inp_img,inp_depth

# data = np.clip(data,0,MAX_DEPTH) / MAX_DEPTH
# for idx in range(ds.length):
#     cv2.imshow("",color_array(data[idx]))
#     cv2.waitKey(1)

# np.save("test_data.npy",data)

batch_size = 256
epochs = 500


data = np.load("test_data.npy")
data = np.stack((data,data,data),-1)
print(data.shape)

labels = np.zeros((ds.length,3))
for idx in range(3):
    labels[:,idx] = (np.copy(ds.angles[:,idx]) - u_reader.joint_limits[idx,0]) /  (u_reader.joint_limits[idx,1] - u_reader.joint_limits[idx,0])

train_labels, test_labels = labels[:TRAIN_SIZE], labels[TRAIN_SIZE:]
train_data, test_data = data[:TRAIN_SIZE], data[TRAIN_SIZE:]

model = keras.models.Sequential()
model.add(layers.Conv2D(8,3,activation='relu',input_shape=(120,120,3)))
model.add(layers.Conv2D(8,3,activation='relu'))
model.add(layers.MaxPool2D(2))
model.add(layers.Conv2D(4,3,activation='relu'))
model.add(layers.Conv2D(4,3,activation='relu'))
model.add(layers.MaxPool2D(2))
model.add(layers.Conv2D(4,3,activation='relu'))
model.add(layers.Conv2D(4,3,activation='relu'))
model.add(layers.MaxPool2D(2))
model.add(layers.Conv2D(4,3,activation='relu'))
model.add(layers.Conv2D(8,3,activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(12,activation='relu'))
model.add(layers.Dense(8,activation='relu'))
model.add(layers.Dense(3,activation='linear'))

print(model.summary())

loss = keras.losses.MeanSquaredLogarithmicError()
optim = keras.optimizers.Adam(lr=.001)
metrics = ['mae','mape','mse']

model.compile(optim,loss,metrics)

model.fit(train_data,train_labels,batch_size,epochs,2)
model.save_weights('weights.h5')
model.save('thisisatest')

model.evaluate(test_data,test_labels,batch_size,2)

# model = keras.models.load_model('thisisatest')

out = model.predict(data,batch_size,2)
for idx in range(3):
    out[:,idx] = (out[:,idx] * (u_reader.joint_limits[idx,1] - u_reader.joint_limits[idx,0]) + u_reader.joint_limits[idx,0])

new = np.zeros((ds.length,6))
new[:,:3] = out

g = Grapher('SLU',new,np.copy(ds.angles))
g.plot()