from robotpose.simulation.render import DatasetRenderer
import robotpose
import numpy as np
from robotpose import Dataset, Grapher
import cv2
from robotpose.prediction.analysis import JointDistance
import PySimpleGUI as sg
import re


import logging, os
# Disable OpenGL and Tensorflow info messages (get annoying)
logging.getLogger("OpenGL.arrays.arraydatatype").setLevel(logging.WARNING)
logging.getLogger("OpenGL.acceleratesupport").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf


#file = sg.PopupGetFile('Select Prediction File','Prediction File Selection',file_types=(("NPY Files", "*.npy"), ),initial_folder=os.getcwd())
file = 'synth_test_.npy'

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


indicies = np.argsort(angles[...,IDX_TO_USE])

out = np.sort(indicies)
# print(out)

g = Grapher('SLU',preds[indicies],angles[indicies])
g.plot(20)

diff = np.abs(preds - angles)

j = JointDistance()
j.plot(preds[indicies],angles[indicies],.25)


#print(diff)

#print((diff[:,0] > np.percentile(diff[:,IDX_TO_USE],PERCENTILE_TO_SHOW)))


# idxs = np.where(diff[:,IDX_TO_USE] > np.percentile(diff[:,IDX_TO_USE],PERCENTILE_TO_SHOW))
# imgs = np.copy(ds.og_img[idxs])
# preds, diff = preds[idxs], diff[idxs]

# r = DatasetRenderer(dataset)
# r.setMaxParts(6)

# for idx in range(len(imgs)):
#     r.setJointAngles(preds[idx])
#     c,d = r.render()
#     out = cv2.addWeighted(imgs[idx],.5,c,.5,0)
#     print("")
#     print(idxs[idx])
#     print([f"{x:0.3f}" for x in (ds.angles[idx] *180 / np.pi)])
#     print([f"{x:0.3f}" for x in (preds[idx] *180 / np.pi)])
#     cv2.imshow("",out)
#     cv2.waitKey(0)