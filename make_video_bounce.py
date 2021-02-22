import os
import cv2
from robotpose import paths as p



cap = cv2.VideoCapture(p.video.replace(".avi","_overlay.avi"))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(p.video.replace(".avi","_overlay_rebound.avi"),fourcc, 30, (640*2,480))

frames = []

ret, frame = cap.read()
while ret:
    frames.append(frame)
    ret, frame = cap.read()

rev = frames.copy()
rev.reverse()

full = frames + rev

for frame in full:
    out.write(frame)

cap.release()
out.release()
