import os
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from tqdm import tqdm
import pickle

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
    return a




def parsePLYasPoints(path):
    # Read file
    with open(path, 'r') as file:
        lines = file.readlines()

    # Read through header
    in_data = False
    while not in_data:

        if 'element vertex' in lines[0]:
            verticies = int(lines[0].replace('element vertex',''))

        if 'end_header' in lines[0]:
            in_data = True
        lines.pop(0)

    # Copy camera intrinisics
    intrin = makeIntrinsics()

    # Extract vertex info
    vert = []
    while len(vert) < verticies:
        string = lines.pop(0)
        data = list(map(float, string.split(' ')[:-1]))
        x, y = rs.rs2_project_point_to_pixel(intrin, data)
        dictionary = {
            'Px':x,
            'Py':y,
            'X': data[0],
            'Y': data[1],
            'Z': data[2]
        }
        vert.append(dictionary)
        lines.pop(0)
        lines.pop(0)
    
    return vert

            
def parsePLYs(path_to_ply, save_path):
    plys = []
    for file in tqdm(os.listdir(path_to_ply)):
        plys.append(parsePLYasPoints(os.path.join(path_to_ply,file)))
    
    if '.pyc' not in save_path:
        save_path = os.path.join(save_path,'ply_data.pyc')

    with open(save_path,'wb') as file:
        pickle.dump(plys,file)

def readBin(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


