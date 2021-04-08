# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from imgaug.augmenters import color
from deepposekit.models import load_model
from deepposekit.io import VideoReader
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from robotpose.utils import *
import pickle
from robotpose import paths as p
from robotpose.dataset import Dataset
from robotpose.utils import reject_outliers_iqr
from robotpose.turbo_colormap import color_array

from robotpose.angle_prediction import Predictor
import sys

setMemoryGrowth()

# Load dataset
ds = Dataset('set0','B')

# Read in Actual angles from JSONs to compare predicted angles to
S_angles = ds.angles[:,0]
L_angles = ds.angles[:,1]
U_angles = ds.angles[:,2]
B_angles = ds.angles[:,4]

# Load model, make predictions
model = load_model(os.path.join(os.getcwd(),r'models\set10__B__CutMobilenet.h5'))
reader = VideoReader(ds.seg_vid_path)
predictions = model.predict(reader)

# np.save('set6_output.npy',predToXYZ(predictions, ds.ply))
# print("Predictions saved")

pred_dict = predToDictList_new(predictions)
pred_dict_xyz = predToXYZdict_new(pred_dict, ds.pointmaps)

# Load video capture and make output
cap = cv2.VideoCapture(ds.seg_vid_path)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(p.VIDEO.replace(".avi","_overlay.avi"),fourcc, 20, (ds.seg_resolution[1]*2,ds.seg_resolution[0]))
#out = cv2.VideoWriter(p.VIDEO.replace(".avi","_overlay.avi"),fourcc, 12.5, (ds.seg_resolution[1],ds.seg_resolution[0]))

# Init predicted angle lists
S_pred = []
L_pred = []
U_pred = []
B_pred = []

ret, image = cap.read()

frame_height = image.shape[0]
frame_width = image.shape[1]


tim = Predictor('B')


i = 0
while ret:
    over = np.zeros((ds.seg_resolution[0],ds.seg_resolution[1],3),dtype=np.uint8)
    coord_dict = pred_dict_xyz[i]

    tim.load(predictions[i], ds.pointmaps[i])
    pred = tim.predict()

    L_pred_ang = pred['L']
    U_pred_ang = pred['U']

    # Append to lists
    S_pred.append(0)
    L_pred.append(L_pred_ang)
    U_pred.append(U_pred_ang)
    #B_pred.append(B_pred_ang)

    # Put depth info on overlay
    over = color_array(ds.pointmaps[i,:,:,2], .1)
    #Visualize lines
    viz(image, over, predictions[i])

    dual = np.zeros((frame_height,frame_width*2,3),dtype=np.uint8)
    dual[:,0:frame_width] = image
    dual[:,frame_width:frame_width*2] = over

    out.write(dual)
    cv2.imshow("Angle Predictions",dual)
    cv2.waitKey(1)
    i+=1
    ret, image = cap.read()

cv2.destroyAllWindows()
cap.release()
out.release()


"""
Plotting Angles
"""

# Convert everything from radians to degrees
S_act = L_act = U_act = B_act = None
for a, b in zip(["S_act","L_act","U_act","S_pred","L_pred","U_pred"],["S_angles","L_angles","U_angles","S_pred","L_pred","U_pred"]):
    globals()[a] = np.degrees(globals()[b])


# Make Subplots
fig, axs = plt.subplots(3,3)

# Plot Raw Angles
for idx, act, pred, label in zip(range(3),["S_act","L_act","U_act",],["S_pred","L_pred","U_pred"],["S","L","U"]):
    axs[idx,0].set_title(f'Raw {label} Angle')
    axs[idx,0].plot(globals()[act])
    axs[idx,0].plot(globals()[pred])


# Offset Angles
S_offset = np.add(np.mean(np.subtract(S_act,S_pred)),S_pred)
L_offset = np.add(np.mean(np.subtract(L_act,L_pred)),L_pred)
U_offset = np.add(np.mean(np.subtract(U_act,U_pred)),U_pred)
#B_offset = np.add(np.mean(np.subtract(B_act,B_pred)),B_pred)

for idx, act, offset, label in zip(range(4),["S_act","L_act","U_act"],["S_offset","L_offset","U_offset"],["S","L","U"]):
    axs[idx,1].set_title(f'Offset {label} Angle')
    axs[idx,1].plot(globals()[act])
    axs[idx,1].plot(globals()[offset])



#Residuals
S_err = np.subtract(S_offset, S_act)
L_err = np.subtract(L_offset, L_act)
U_err = np.subtract(U_offset, U_act)
#B_err = np.subtract(B_offset, B_act)

zeros_err = np.zeros(L_act.shape)

for idx, err, label in zip(range(4),["S_err","L_err","U_err"],["S","L","U"]):
    axs[idx,2].set_title(f'Angle {label} Error')
    axs[idx,2].plot(zeros_err)
    axs[idx,2].plot(globals()[err])


# Determine average errors
avg_S_err = np.mean(np.abs(S_err))
avg_L_err = np.mean(np.abs(L_err))
avg_U_err = np.mean(np.abs(U_err))
#avg_B_err = np.mean(np.abs(B_err))
S_err_std = np.std(np.abs(S_err))
L_err_std = np.std(np.abs(L_err))
U_err_std = np.std(np.abs(U_err))
#B_err_std = np.std(np.abs(B_err))

print("\nAvg Error (deg):")
print(f"\tS: {avg_S_err:.2f}\n\tL: {avg_L_err:.2f}\n\tU: {avg_U_err:.2f}")
print("Stdev (deg):")
print(f"\tS: {S_err_std:.2f}\n\tL: {L_err_std:.2f}\n\tU: {U_err_std:.2f}")


# Determine average errors without outliers
avg_S_err = np.mean(reject_outliers_iqr(np.abs(S_err)))
avg_L_err = np.mean(reject_outliers_iqr(np.abs(L_err)))
avg_U_err = np.mean(reject_outliers_iqr(np.abs(U_err)))
#avg_B_err = np.mean(np.abs(B_err))
S_err_std = np.std(reject_outliers_iqr(np.abs(S_err)))
L_err_std = np.std(reject_outliers_iqr(np.abs(L_err)))
U_err_std = np.std(reject_outliers_iqr(np.abs(U_err)))
#B_err_std = np.std(np.abs(B_err))

print("\nOutliers Removed:")
print("Avg Error (deg):")
print(f"\tS: {avg_S_err:.2f}\n\tL: {avg_L_err:.2f}\n\tU: {avg_U_err:.2f}")
print("Stdev (deg):")
print(f"\tS: {S_err_std:.2f}\n\tL: {L_err_std:.2f}\n\tU: {U_err_std:.2f}")





plt.show()