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


# Read in Actual angles from JSONs
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

    viz_points(ply_data[i], over)
    
    # Predict L
    L_pred_ang = vecXZang(coord_dict['L'], coord_dict['U'])

    # if i < 62:
    #     L_pred_ang = np.pi - L_pred_ang

    # Predict U
    U_pred_ang =  vecXZang(coord_dict['L'], coord_dict['U']) - vecXZang(coord_dict['B'], coord_dict['R'])

    # if i > 64:
    #     U_pred_ang = -1*np.pi+U_pred_ang


    # Predict B
    B_pred_ang = vecXZang(coord_dict['T'], coord_dict['B']) - vecXZang(coord_dict['B'], coord_dict['R'])

    # Append to lists
    L_pred.append(L_pred_ang)
    U_pred.append(U_pred_ang)
    B_pred.append(B_pred_ang)

    #Visualize lines
    viz(image, over, predictions[i])

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

fig, axs = plt.subplots(3,3)
axs[0,0].plot(toDeg(L_angles))
axs[0,0].plot(toDeg(L_pred))
axs[0,0].set_title('Raw L Angle')
axs[1,0].plot(toDeg(U_angles))
axs[1,0].plot(toDeg(U_pred))
axs[1,0].set_title('Raw U Angle')
axs[2,0].plot(toDeg(B_angles))
axs[2,0].plot(toDeg(B_pred))
axs[2,0].set_title('Raw B Angle')

axs[0,1].plot(toDeg(L_angles))
axs[0,1].plot(toDeg(np.add(np.mean(np.subtract(L_angles,L_pred)),L_pred)))
axs[0,1].set_title('Offset L Angle')
axs[1,1].plot(toDeg(U_angles))
axs[1,1].plot(toDeg(np.add(np.mean(np.subtract(U_angles,U_pred)),U_pred)))
axs[1,1].set_title('Offset U Angle')
axs[2,1].plot(toDeg(B_angles))
axs[2,1].plot(toDeg(np.add(np.mean(np.subtract(B_angles,B_pred)),B_pred)))
axs[2,1].set_title('Offset B Angle')

axs[0,2].plot(np.multiply(toDeg(L_angles),0))
axs[0,2].plot(toDeg(np.subtract(L_angles,np.add(np.mean(np.subtract(L_angles,L_pred)),L_pred))))
axs[0,2].set_title('Offset L Err')
axs[1,2].plot(toDeg(np.multiply(toDeg(L_angles),0)))
axs[1,2].plot(toDeg(np.subtract(U_angles,np.add(np.mean(np.subtract(U_angles,U_pred)),U_pred))))
axs[1,2].set_title('Offset U Err')
axs[2,2].plot(toDeg(np.multiply(toDeg(L_angles),0)))
axs[2,2].plot(toDeg(np.subtract(B_angles,np.add(np.mean(np.subtract(B_angles,B_pred)),B_pred))))
axs[2,2].set_title('Offset B Err')

plt.show()