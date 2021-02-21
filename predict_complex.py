from deepposekit.models import load_model
from deepposekit.io import VideoReader
import cv2
import matplotlib.pyplot as plt
from deepposekit.io import TrainingGenerator, DataGenerator
import numpy as np
import pyrealsense2 as rs
from robotpose.utils import *


vid_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\video.avi"
image_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\2d"
skeleton_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\mult_skeleton.csv"
model_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\model_LEAP_mult.h5"

L_angles = readLinkXData(1)
U_angles = readLinkXData(2)
B_angles = readLinkXData(4)

model = load_model(model_path)
reader = VideoReader(vid_path)
predictions = model.predict(reader)
pred_dict = predToDictList(predictions)

cap = cv2.VideoCapture(vid_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(vid_path.replace(".avi","_overlay.avi"),fourcc, 12.5, (640*2,480))

L_pred = []
U_pred = []
B_pred = []

ret, image = cap.read()
i = 0
while ret:
    over = np.zeros((480,640,3),dtype=np.uint8)
    coord_dict = pred_dict[i]
    
    # Predict L
    Lx = coord_dict['U'][0] - coord_dict['L'][0]
    Ly = coord_dict['U'][1] - coord_dict['L'][1]
    L_pred_ang = angle(Lx, Ly, (.5,-100))

    # Predict U
    Ux = coord_dict['R'][0] - coord_dict['B'][0]
    Uy = coord_dict['R'][1] - coord_dict['B'][1]
    U_pred_ang = L_pred_ang - angle(Ux,Uy)

    # Predict B
    Bx = coord_dict['B'][0] - coord_dict['T'][0]
    By = coord_dict['B'][1] - coord_dict['T'][1]
    B_pred_ang = angle(Bx,By) - angle(Ux,Uy)

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
axs[1,0].plot(toDeg(U_angles))
axs[1,0].plot(toDeg(U_pred))
axs[2,0].plot(toDeg(B_angles))
axs[2,0].plot(toDeg(B_pred))

axs[0,1].plot(toDeg(L_angles))
axs[0,1].plot(toDeg(np.add(np.mean(np.subtract(L_angles,L_pred)),L_pred)))
axs[1,1].plot(toDeg(U_angles))
axs[1,1].plot(toDeg(np.add(np.mean(np.subtract(U_angles,U_pred)),U_pred)))
axs[2,1].plot(toDeg(B_angles))
axs[2,1].plot(toDeg(np.add(np.mean(np.subtract(B_angles,B_pred)),B_pred)))

axs[0,2].plot(np.multiply(toDeg(L_angles),0))
axs[0,2].plot(toDeg(np.subtract(L_angles,np.add(np.mean(np.subtract(L_angles,L_pred)),L_pred))))
axs[1,2].plot(toDeg(np.multiply(toDeg(L_angles),0)))
axs[1,2].plot(toDeg(np.subtract(U_angles,np.add(np.mean(np.subtract(U_angles,U_pred)),U_pred))))
axs[2,2].plot(toDeg(np.multiply(toDeg(L_angles),0)))
axs[2,2].plot(toDeg(np.subtract(B_angles,np.add(np.mean(np.subtract(B_angles,B_pred)),B_pred))))


plt.show()