import os
import cv2
from robotpose import paths as p
import imageio
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
args = parser.parse_args()

cap = cv2.VideoCapture(args.path)


frames = []

ret, frame = cap.read()
while ret:
    frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    ret, frame = cap.read()
cap.release()

rev = frames.copy()
rev.reverse()

full = frames + rev
full = np.asarray(full)

imageio.mimsave('assets/a.gif', full, duration = .02)
print("Done")





