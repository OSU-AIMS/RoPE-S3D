import os
import cv2


image_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\2d"
vid_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\video.avi"
ds_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\ds.h5"
skeleton_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\skeleton.csv"
log_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\log.h5"
model_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\model_dense.h5"

writer = None

for file in os.listdir(image_path):
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(vid_path,fourcc, 20.0, (640,480))

    img = cv2.imread(os.path.join(image_path,file))
    writer.write(img)

writer.release()