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


# Load dataset
ds = Dataset('set5','B',primary="rm")
print(ds.resolution)

# Read in Actual angles from JSONs to compare predicted angles to
S_angles = ds.angles[:,0]
L_angles = ds.angles[:,1]
U_angles = ds.angles[:,2]
B_angles = ds.angles[:,4]

# Load model, make predictions
model = load_model(r'C:\Users\exley\OneDrive\Documents\GitHub\DeepPoseRobot\models\deeppose_B_CutMobilenet_1.h5')
reader = VideoReader(ds.vid_path)
predictions = model.predict(reader)


# Load video capture and make output
cap = ds.vid
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(p.video.replace(".avi","_overlay.avi"),fourcc, 12.5, (ds.resolution[1]*2,ds.resolution[0]))

ret, image = cap.read()

frame_height = image.shape[0]
frame_width = image.shape[1]

i = 0
while ret:
    over = np.zeros((ds.resolution[0],ds.resolution[1],3),dtype=np.uint8)

    # Put depth info on overlay
    #vizDepth(ds.ply[i], over)
    #Visualize lines
    viz(image, over, predictions[i])

    dual = np.zeros((frame_height,frame_width*2,3),dtype=np.uint8)
    dual[:,0:frame_width] = image
    dual[:,frame_width:frame_width*2] = over

    out.write(dual)
    cv2.imshow("test",dual)
    cv2.waitKey(1)
    i+=1
    ret, image = cap.read()

cv2.destroyAllWindows()
cap.release()
out.release()

