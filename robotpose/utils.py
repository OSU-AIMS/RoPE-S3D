import os
import json
import numpy as np
import cv2
import pyrealsense2 as rs

def readJsonData(json_path = "C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\json"):
    data = []

    for file in os.listdir(json_path):
        file_path = os.path.join(json_path,file)
        with open(file_path) as f:
            d = json.load(f)

        d = d['objects'][0]['joint_angles']

        data.append(d)

    return data


def readLinkXData(link):
    data = readJsonData()
    angles = []
    for entry in data:
        ang = entry[link]['angle']
        angles.append(ang)

    return angles


def toDeg(arr):
    return np.multiply(arr, 180/np.pi)


def angle(x, y, lims=None):
    ang = np.arctan(y/x)
    if x < 0:
        ang += np.pi

    if lims is not None:
        if ang > max(lims):
            ang -= 2 * np.pi
        elif ang < min(lims):
            ang += 2* np.pi
    
    return ang


def predToDictList(preds):
    out = []
    for p in preds:
        out.append({'L':p[0],
                    'midL':p[1],
                    'U':p[2],
                    'R':p[3],
                    'B':p[4],
                    'T':p[5]})
    return out
    

def viz(image, over, frame_data):
    last = None
    for p in frame_data:
        x = int(p[0])
        y = int(p[1])

        if last is not None:
            image = cv2.line(image, (x,y), last, color=(255, 0, 0), thickness=3)
            over = cv2.line(over, (x,y), last, color=(255, 0, 0), thickness=3)

        image = cv2.circle(image, (x,y), radius=4, color=(0, 0, 255), thickness=-1)
        over = cv2.circle(over, (x,y), radius=4, color=(0, 0, 255), thickness=-1)
        last = (x,y)


def makeIntrinsics(resolution = (640,480), pp= (320.503,237.288), f=(611.528,611.528), coeffs=[0,0,0,0,0]):
    a = rs.intrinsics()
    a.width = max(resolution)
    a.height = min(resolution)
    a.ppx = pp[0]
    a.ppy = pp[1]
    a.fx = f[0]
    a.fy = f[1]
    a.coeffs = coeffs
