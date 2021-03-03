import os
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from tqdm import tqdm
import pickle
from robotpose import paths as p
import open3d as o3d


def limitMemory():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)




def XYangle(x, y, lims=None):
    """
    Returns the angle between the input point and the +x axis 
    """
    # Get angle
    ang = np.arctan(y/x)

    # If in Quad II/III, make the angle obtuse 
    if x < 0:
        ang += np.pi

    # Apply any custom limits to the angle
    if lims is not None:
        if ang > max(lims):
            ang -= 2 * np.pi
        elif ang < min(lims):
            ang += 2* np.pi
    
    return ang


def XYZangle(start, end, lims = None):
    """
    Rotates vector from start to end into the XY plane and uses XY angle to calculate angle
    """
    # Find link vector from start and end points
    vec = np.subtract(end, start)

    # Rotate the reference XY plane to contain the vector
    rotated_y = vec[1]
    rotated_x = np.sqrt(vec[0] ** 2 + vec[2] ** 2) * abs(vec[0]) / vec[0]

    # Find 2D angle to X axis
    return XYangle(rotated_x, rotated_y, lims)



def predToDictList(preds):
    """
    Takes predictions from DeepPoseKit as list and translates into a dictionary of points
    """
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
    """
    Draws a pose overlay on the image given prediction point data
    """
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


def makeIntrinsics(res = 1280):
    if res == 1280:
        return intrin(resolution = (1280,720), pp= (638.391,361.493), f=(905.23, 904.858), coeffs=[0,0,0,0,0])
    elif res == 640:
        return intrin(resolution = (640,480), pp= (320.503,237.288), f=(611.528,611.528), coeffs=[0,0,0,0,0])
    elif res == 1281:
        return intrin(resolution = (1280,720), pp= (639.459,359.856), f=(635.956, 635.956), coeffs=[0,0,0,0,0])



def intrin(resolution = (640,480), pp= (320.503,237.288), f=(611.528,611.528), coeffs=[0,0,0,0,0]):
    """
    Makes psuedo-intrinsics for the realsense camera used.
    """
    a = rs.intrinsics()
    a.width = max(resolution)
    a.height = min(resolution)
    a.ppx = pp[0]
    a.ppy = pp[1]
    a.fx = f[0]
    a.fy = f[1]
    a.coeffs = coeffs
    return a



def predToXYZ(dict_list, ply_data):
    """
    Using the complete list of dictionaries and 3D data, find the XYZ coords of each keypoint 
    """
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

def vizDepth(ply_frame_data, image):
    """
    Overlays the depth information given on an image
    """
    intrin = makeIntrinsics()
    for pt in ply_frame_data:
        x, y = rs.rs2_project_point_to_pixel(intrin, pt[2:5])
        x = int(x)
        y = int(y)
        g = int(np.interp(pt[4],[-1.3,-.9],[0,255]))
        r = 255-2*g
        image = cv2.circle(image, (x,y), radius=0, color=(0,g,r), thickness=-1)









"""
DEPRECATED FUNCTIONS
These functions are in the process of being replaced by the dataset class
"""






def parsePLYasPoints(path):
    """
    Reads in data from a PLY file, extracting only the verticies
    Returns data as a dictionary
    """
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
        # Make dictionary of data
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
    """
    Parses a folder of PLY files into a single pickled file
    """
    plys = []

    ply_files = [x for x in os.listdir(path_to_ply) if x.endswith('.ply')]

    for file in tqdm(ply_files,desc="Parsing PLY data"):
        plys.append(parsePLYasPoints(os.path.join(path_to_ply,file)))
    
    if '.pyc' not in save_path:
        save_path = os.path.join(save_path,'ply_data.pyc')

    with open(save_path,'wb') as file:
        pickle.dump(plys,file)




def readBinToArrs(path):
    """
    Reads pickled PLY files into numpy arrays that can be efficently modified
    """
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


def readJsonData(json_path = p.json):
    """
    Reads angle data from all JSON files and returns as list of dicts 
    """
    data = []

    jsons = [x for x in os.listdir(json_path) if x.endswith('.json')]

    for file in jsons:
        file_path = os.path.join(json_path,file)
        with open(file_path) as f:
            d = json.load(f)

        d = d['objects'][0]['joint_angles']

        data.append(d)

    return data


def readLinkXData(link):
    """
    Reads JSON files and returns an array of the values for a specific joint
    """
    data = readJsonData()
    angles = []
    for entry in data:
        ang = entry[link]['angle']
        angles.append(ang)

    return angles


def makeVideo(image_path = p.image, vid_path = p.video):
    """
    Combines folder of images into a video
    """
    writer = None

    images = [x for x in os.listdir(image_path) if x.endswith('.jpg')]
    images += [x for x in os.listdir(image_path) if x.endswith('.png')]

    for file in os.listdir(image_path):
        img = cv2.imread(os.path.join(image_path,file))

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(vid_path,fourcc, 20.0, (img.shape[1],img.shape[0]))

        
        writer.write(img)

    writer.release()


def renamePNG(path):
    imgs = [x for x in os.listdir(path) if x.endswith('.png')]
    for img in imgs:
        os.rename(os.path.join(path,img),os.path.join(path,img.replace('.png','_rm.png')))