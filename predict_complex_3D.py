from deepposekit.models import load_model
from deepposekit.io import VideoReader
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyrealsense2 as rs
from robotpose.utils import *
import pickle
from robotpose import paths as p


# Compile PLY data if not already complied
if not os.path.isfile(p.ply_data):
    parsePLYs()

# Read ply data
ply_data = readBinToArrs(p.ply_data)


# Read in Actual angles from JSONs to compare predicted angles to
L_angles = readLinkXData(1)
U_angles = readLinkXData(2)
B_angles = readLinkXData(4)

# Load model, make predictions
model = load_model(p.model_mult)
reader = VideoReader(p.video)
predictions = model.predict(reader)
pred_dict = predToDictList(predictions)
pred_dict_xyz = dictPixToXYZ(pred_dict, ply_data)

# Load video capture and make output
cap = cv2.VideoCapture(p.video)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(p.video.replace(".avi","_overlay.avi"),fourcc, 12.5, (640*2,480))

# Init predicted angle lists
L_pred = []
U_pred = []
B_pred = []

ret, image = cap.read()
i = 0
while ret:
    over = np.zeros((480,640,3),dtype=np.uint8)
    coord_dict = pred_dict_xyz[i]
    
    # Predict L
    L_pred_ang = vecXZangNew(coord_dict['L'], coord_dict['U'])

    BR_ang = vecXZangNew(coord_dict['B'], coord_dict['R'],(.5,-10))

    # Predict U
    U_pred_ang = L_pred_ang - BR_ang
    # Predict B
    B_pred_ang = vecXZangNew(coord_dict['T'], coord_dict['B'],(1,-10)) - BR_ang

    # Append to lists
    L_pred.append(L_pred_ang)
    U_pred.append(U_pred_ang)
    B_pred.append(B_pred_ang)

    #Visualize lines
    viz(image, over, predictions[i])
    # Put depth info on overlay
    viz_points(ply_data[i], over)

    dual = np.zeros((480,640*2,3),dtype=np.uint8)
    dual[:,0:640] = image
    dual[:,640:640*2] = over

    out.write(dual)
    cv2.imshow("test",dual)
    cv2.waitKey(1)
    i+=1
    ret, image = cap.read()

cv2.destroyAllWindows()
cap.release()
out.release()


"""
Plotting Angles
"""

fig, axs = plt.subplots(3,3)

# Convert Radians to degrees
L_act = toDeg(L_angles)
U_act = toDeg(U_angles)
B_act = toDeg(B_angles)

L_pred = toDeg(L_pred)
U_pred = toDeg(U_pred)
B_pred = toDeg(B_pred)

# Raw Angles
axs[0,0].set_title('Raw L Angle')
axs[0,0].plot(L_act)
axs[0,0].plot(L_pred)

axs[1,0].set_title('Raw U Angle')
axs[1,0].plot(U_act)
axs[1,0].plot(U_pred)

axs[2,0].set_title('Raw B Angle')
axs[2,0].plot(B_act)
axs[2,0].plot(B_pred)


# Offset Angles
L_offset = np.add(np.mean(np.subtract(L_act,L_pred)),L_pred)
U_offset = np.add(np.mean(np.subtract(U_act,U_pred)),U_pred)
B_offset = np.add(np.mean(np.subtract(B_act,B_pred)),B_pred)

axs[0,1].set_title('Offset L Angle')
axs[0,1].plot(L_act)
axs[0,1].plot(L_offset)

axs[1,1].set_title('Offset U Angle')
axs[1,1].plot(U_act)
axs[1,1].plot(U_offset)

axs[2,1].set_title('Offset B Angle')
axs[2,1].plot(B_act)
axs[2,1].plot(B_offset)

#Residuals
L_err = np.subtract(L_offset, L_act)
U_err = np.subtract(U_offset, U_act)
B_err = np.subtract(B_offset, B_act)

axs[0,2].set_title('Offset L Err')
axs[0,2].plot(np.zeros(L_act.shape))
axs[0,2].plot(L_err)

axs[1,2].set_title('Offset U Err')
axs[1,2].plot(np.zeros(U_act.shape))
axs[1,2].plot(U_err)

axs[2,2].set_title('Offset B Err')
axs[2,2].plot(np.zeros(B_act.shape))
axs[2,2].plot(B_err)


# Determine average errors
avg_L_err = np.mean(np.abs(L_err))
avg_U_err = np.mean(np.abs(U_err))
avg_B_err = np.mean(np.abs(B_err))
L_err_std = np.std(np.abs(L_err))
U_err_std = np.std(np.abs(U_err))
B_err_std = np.std(np.abs(B_err))

print("Average error in degrees (after an offset is applied):")
print(f"\tL: {avg_L_err}\n\tU: {avg_U_err}\n\tB: {avg_B_err}")
print("Stdev of error in degrees (after an offset is applied):")
print(f"\tL: {L_err_std}\n\tU: {U_err_std}\n\tB: {B_err_std}")

plt.show()