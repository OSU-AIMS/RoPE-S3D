# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley


import os

import numpy as np
import cv2
import pyrealsense2 as rs
from tqdm import tqdm
from .paths import Paths as p
import time
from .projection import makeIntrinsics
from .turbo_colormap import normalize_and_interpolate

import multiprocessing as mp


def setMemoryGrowth():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)


def workerCount():
    cpu_count = mp.cpu_count()
    return int(min(cpu_count - 2, .75 * cpu_count))


def expandRegion(image, size, iterations = 1):
    kern = np.ones((size,size), dtype=np.uint8)
    return cv2.dilate(image, kern, iterations = iterations)





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


def predToDictList_new(preds):
    """
    Takes predictions from DeepPoseKit as list and translates into a dictionary of points
    """
    out = []
    for p in preds:
        out.append({'base':p[0],
                    'L':p[1],
                    'U':p[2],
                    'R':p[3],
                    'B':p[4]})
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



def predToXYZdict(dict_list, ply_data):
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


def predToXYZdict_new(dict_list, ply_data):
    """
    Using the complete list of dictionaries and 3D data, find the XYZ coords of each keypoint 
    """
    ply_data = np.asarray(ply_data)
    out = []
    for d, idx in tqdm(zip(dict_list,range(len(dict_list)))):
        data = ply_data[idx]
        out_dict = {}
        for key, value in zip(d.keys(), d.values()):
            px = int(value[0])
            py = int(value[1])
            out_dict[key] = tuple(data[py,px])
        
        out.append(out_dict)

    return out


def predToXYZ(preds, ply_data):
    ply_data = np.asarray(ply_data)
    # Make sure there are the same number of frame predictions as ply frames
    assert len(preds) == ply_data.shape[0]

    # Create output array
    out = np.zeros((ply_data.shape[0], len(preds[0]), 3))

    # Go through each frame
    for pred, ply, idx in zip(preds, ply_data, range(len(preds))):
        x_list = ply[:,0]
        y_list = ply[:,1]

        # Go through each point in the frame
        for point, sub_idx in zip(pred, range(len(pred))):
            px, py = point[0:2]

            # Find closest point
            dist = np.sqrt( np.square( x_list - px ) + np.square( y_list - py ) )
            min_idx = dist.argmin()
            out[idx,sub_idx] = tuple(ply[min_idx,2:5])

    return out




def vizDepth_new(ply_frame_data, image):
    """
    Overlays the depth information given on an image
    """
    z_min, z_max = outlier_min_max(ply_frame_data[:,4], iqr_mult=3.0)
    idx_arr = ply_frame_data[:,0:2].astype(int)
    #print(f"Min: {np.min(ply_frame_data[:,4])}\t{z_min}\nMax: {np.max(ply_frame_data[:,4])}\t{z_max}\n")
    for idx in range(len(ply_frame_data)):
        color = normalize_and_interpolate(ply_frame_data[idx,4], z_min, z_max)
        color.reverse() # Switch from BGR/RGB
        image = cv2.circle(image, (idx_arr[idx,0],idx_arr[idx,1]), radius=1, color=color, thickness=-1)

    # plt.hist(ply_frame_data[:,4], bins=200)
    # plt.yscale('log')
    # plt.show()

    return image

def reject_outliers_std(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def outlier_min_max(data, iqr_mult = 1.5):
    data = reject_outliers_iqr(data, iqr_mult)
    return np.min(data), np.max(data)

def reject_outliers_iqr(data, iqr_mult = 1.5):
    percentiles = np.percentile(data, [75, 25])
    iqr = np.subtract(*percentiles)
    max = percentiles[0] + iqr_mult * iqr
    min = percentiles[1] - iqr_mult * iqr
    data = data[data >= min]
    data = data[data <= max]
    return data



class Timer():
    def __init__(self):
        self.data = {}
        self.start()
    
    def start(self):
        self.start_time = time.time()

    def split(self,split_name):
        try:
            self.data[split_name] += time.time() - self.start_time
        except KeyError:
            self.data[split_name] = time.time() - self.start_time
        self.start()

    def __repr__(self):
        out = "Times:"
        tot = 0
        for item in self.data:
            tot += self.data[item]
        for item in self.data:
            out += f"\n\t{item}: {self.data[item]:.3f}s {(self.data[item] * 100/tot):.2f}%"
        return out




def vizDepth(ply_frame_data, image, x_crop):
    """
    Overlays the depth information given on an image
    """
    intrin = makeIntrinsics()
    for pt in ply_frame_data:
        x, y = rs.rs2_project_point_to_pixel(intrin, pt[2:5])
        x = int(x)-x_crop
        y = int(y)
        g = int(np.interp(pt[4],[-1.3,-.9],[0,255]))
        r = 255-2*g
        image = cv2.circle(image, (x,y), radius=0, color=(0,g,r), thickness=-1)






"""
DEPRECATED FUNCTIONS
These functions are in the process of being replaced by the dataset class
"""


def vizDepth_old(ply_frame_data, image):
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



def renamePNG(path):
    imgs = [x for x in os.listdir(path) if x.endswith('.png')]
    for img in imgs:
        os.rename(os.path.join(path,img),os.path.join(path,img.replace('.png','_rm.png')))