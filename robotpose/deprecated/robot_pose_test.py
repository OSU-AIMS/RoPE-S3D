from deepposekit.models import load_model
from deepposekit.io import VideoReader
import cv2
import matplotlib.pyplot as plt
from deepposekit.io import TrainingGenerator, DataGenerator
import numpy as np
from robotpose.utils import readLinkXData

vid_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\video.avi"
image_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\2d"
ds_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\ds.h5"
skeleton_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\skeleton.csv"
log_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\log.h5"
#model_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\model_dense.h5"
model_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\model_LEAP_mult.h5"

U_angles = readLinkXData(2)

model = load_model(model_path)
reader = VideoReader(vid_path)
predictions = model.predict(reader)

cap = cv2.VideoCapture(vid_path)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(vid_path.replace(".avi","_overlay.avi"),fourcc, 15.0, (640*2,480))

detected_angles = []

ret, image = cap.read()
i = 0
while ret:
    over = np.zeros((480,640,3),dtype=np.uint8)
    last = None

    L_pos = predictions[i][0][:2]
    U_pos = predictions[i][1][:2]
    B_pos = predictions[i][2][:2]

    LU = (L_pos[0]-U_pos[0])**2 + (L_pos[1]-U_pos[1])**2
    UB = (B_pos[0]-U_pos[0])**2 + (B_pos[1]-U_pos[1])**2
    LB = (L_pos[0]-B_pos[0])**2 + (L_pos[1]-B_pos[1])**2

    pred_angle = np.arccos(((LU + UB - LB)/(2* (LU*UB)**0.5)))

    m = (U_pos[1] - L_pos[1]) / (U_pos[0] - L_pos[0])

    crit = m*(B_pos[0]-U_pos[0]) + U_pos[1]

    CaseA = (m < 0 and B_pos[1] < m*(B_pos[0]-U_pos[0]) + U_pos[1])
    CaseB = (m > 0 and B_pos[1] > m*(B_pos[0]-U_pos[0]) + U_pos[1])

    if CaseA or CaseB:   
        pred_angle = np.pi*2-pred_angle

    detected_angles.append(pred_angle)

    for p in predictions[i]:
        x = int(p[0])
        y = int(p[1])

        if last is not None:
            image = cv2.line(image, (x,y), last, color=(255, 0, 0), thickness=3)
            over = cv2.line(over, (x,y), last, color=(255, 0, 0), thickness=3)

        image = cv2.circle(image, (x,y), radius=4, color=(0, 0, 255), thickness=-1)
        over = cv2.circle(over, (x,y), radius=4, color=(0, 0, 255), thickness=-1)
        last = (x,y)

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

diff = np.subtract(detected_angles,U_angles)
avg_diff = np.mean(diff)
corrected = np.subtract(detected_angles,avg_diff)
new_diff = np.subtract(corrected,U_angles)
new_diff = np.multiply(new_diff,(180/np.pi))

plt.plot(U_angles)
#plt.plot(detected_angles)
plt.plot(corrected)
plt.show()

plt.plot(new_diff)
plt.show()