import os
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from tqdm import tqdm
import pickle
from robotpose import paths as p

def readJsonData(json_path = p.json):
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
        data[0] *= -1
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

            
def parsePLYs(path_to_ply = p.ply, save_path = p.ply_data):
    plys = []
    for file in tqdm(os.listdir(path_to_ply),desc="Reading PLY data"):
        plys.append(parsePLYasPoints(os.path.join(path_to_ply,file)))
    
    if '.pyc' not in save_path:
        save_path = os.path.join(save_path,'ply_data.pyc')

    with open(save_path,'wb') as file:
        pickle.dump(plys,file)




def readBin(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def readBinToArrs(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)

    out = []

    for frame in tqdm(data,desc="Reading Frame 3D Data"):
        frame_data = np.zeros((len(frame),5))
        for point, idx in zip(frame, range(len(frame))):
            frame_data[idx, 0] = point['Px']
            frame_data[idx, 1] = point['Py']
            frame_data[idx, 2] = point['X']
            frame_data[idx, 3] = point['Y']
            frame_data[idx, 4] = point['Z']
        out.append(frame_data)
    
    return out



def unit_vector(vec):
    return vec / np.linalg.norm(vec)


def vecXZang(start, end, x_correct = True, y_correct = True):
    # Find vector and unit vector
    vec = np.subtract(end, start)
    unit = unit_vector(vec)
    # Plane represented by unit vector with no Y component
    pl_vec = vec
    pl_vec[1] = 0
    plane = unit_vector(pl_vec)

    # Find angle
    ang = np.arccos(np.clip(np.dot(unit, plane), -1, 1))

    if vec[0] < 0 and x_correct:
        ang = np.pi - ang

    if vec[1] < 0 and y_correct:
        ang = 2*np.pi - ang

    return ang


def vecXZangNew(start, end, lims = None):
    # Find vector and unit vector
    vec = np.subtract(end, start)

    rotated_y = vec[1]
    rotated_x = np.sqrt(vec[0] ** 2 + vec[2] ** 2) * abs(vec[0]) / vec[0]

    return angle(rotated_x, rotated_y, lims)

def dictPixToXYZ(dict_list, ply_data):
    ply_data = np.asarray(ply_data)
    out = []
    for d, idx in tqdm(zip(dict_list,range(len(dict_list)))):
        data = ply_data[idx]
        x_list = data[:,0]
        y_list = data[:,1]
        out_dict = {}
        for key, value in zip(d.keys(), d.values()):
            px = value[0]
            py = value[1]
            dist = np.sqrt( np.square( x_list - px ) + np.square( y_list - py ) )
            min_idx = dist.argmin()
            out_dict[key] = tuple(data[min_idx,2:5])
        
        out.append(out_dict)

    return out

def viz_points(ply_frame_data, image):
    intrin = makeIntrinsics()

    for pt in ply_frame_data:
        x, y = rs.rs2_project_point_to_pixel(intrin, pt[2:5])
        x = int(x)
        y = int(y)
        image = cv2.circle(image, (x,y), radius=0, color=(0, 255, 0), thickness=-1)