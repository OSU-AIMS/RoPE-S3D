# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley


import pixellib
from pixellib.instance import custom_segmentation
import cv2
import os
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from .utils import Timer, vizDepth_new
from tqdm import tqdm
from . import paths as p
import time
from . import projection as proj


class RobotSegmenter():

    def __init__(self, resolution = (720,800), model_path = os.path.join(p.seg_models,'A.h5'), intrinsics = '1280_720_color'):
        self.master = custom_segmentation()
        self.master.inferConfig(num_classes= 1, class_names= ["BG", "mh5"])
        self.master.load_model(model_path)
        self.crop_resolution = resolution
        self.intrinsics = proj.makeIntrinsics(intrinsics)

    def height(self):
        return self.crop_resolution[0]
    def width(self):
        return self.crop_resolution[1]

    def segmentImage(self, img, ply_path, debug=False):
        # Load image if given path
        if type(img) is str:
            image = cv2.imread(img)
        else:
            image = img

        image = np.asarray(image)
        tmp = np.copy(image)


        # Detect image
        r, output = self.master.segmentImage(tmp, process_frame=True)


        # Get mask and roi
        mask = np.asarray(r['masks'])
        mask = mask[:,:,0]
        roi = r['rois'][0] # Y1,X1,Y2,X2

        init_roi = np.asarray(roi)

        """
        Get ROI to be the same as the crop size
        """
        # Expand ROI up and down
        while roi[2] - roi[0] < self.crop_resolution[0]:
            # Expand Up
            if roi[0] > 0:
                roi[0] -= 1
            
            #Expand Down
            if roi[2] < image.shape[0]:
                roi[2] += 1

        # Make sure ROI is exact crop size needed
        while roi[2] - roi[0] > self.crop_resolution[0]:
            roi[0] += 1


        # Expand ROI left and right
        while roi[3] - roi[1] < self.crop_resolution[1]:
            # Expand Left
            if roi[1] > 0:
                roi[1] -= 1
            
            #Expand Right
            if roi[3] < image.shape[1]:
                roi[3] += 1

        # Make sure ROI is exact crop size needed
        while roi[3] - roi[1] > self.crop_resolution[1]:
            roi[1] += 1

        assert roi[3] - roi[1] == self.crop_resolution[1], "ROI Crop Width Incorrect"
        assert roi[2] - roi[0] == self.crop_resolution[0], "ROI Crop Height Incorrect"


        """
        Mask Modifications

        Usually doesn't segment ~20 pix from bottom
        """

        # Base how far it goes down on how many are around it in an x-pixel radius
        look_up_dist = 27
        look_side_dist = 10 # one way

        for col in range(mask.shape[1]):
                if mask[image.shape[0]-look_up_dist,col]:
                    # Find how many are each side
                    down = np.sum(mask[image.shape[0]-look_up_dist,col-look_side_dist:col+look_side_dist])
                    # Arbitrary calc
                    to_go = round(look_up_dist * down**2 / (look_side_dist*1.5)**2)
                    # Truncate
                    if to_go > look_up_dist:
                        to_go = look_up_dist
                    # Go down so many from row
                    mask[image.shape[0]-look_up_dist:image.shape[0]-look_up_dist+to_go,col] = True



        """
        Crop out PLY data
        """

        # Open PLY
        cloud = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(cloud.points)

        # Invert X Coords
        points[:,0] = points[:,0] * -1

        # Align XYZ points relative to the color camera instead of the depth camera
        points[:,0] -= .0175

        crop_ply_data = []


        # Get pixel location of each point
        points_proj = proj.proj_point_to_pixel(self.intrinsics, points)
        
        if debug:
            ####################
            temp = np.zeros((points_proj.shape[0],5))
            temp[:,0:2] = points_proj
            temp[:,2:5] = points
            temp_show = np.zeros((720,1280,3),dtype=np.uint8)
            cv2.imshow("Before",vizDepth_new(temp,temp_show))
            cv2.waitKey(1)
            ####################


        points_proj_idx = np.zeros(points_proj.shape,dtype=int)
        points_proj_idx[:,0] = np.round(np.clip(points_proj[:,0],0,1279))
        points_proj_idx[:,1] = np.round(np.clip(points_proj[:,1],0,719))

        if debug:
            ####################
            temp = np.zeros((points_proj.shape[0],5))
            temp[:,0:2] = points_proj_idx
            temp[:,2:5] = points
            temp_show = np.zeros((720,1280,3),dtype=np.uint8)
            vizDepth_new(temp,temp_show)
            temp_show = temp_show *.5 + image *.5
            cv2.imshow("Before_overlay",temp_show.astype(np.uint8))
            cv2.waitKey(1)
            ####################


        for row in range(points_proj.shape[0]):
            if mask[points_proj_idx[row,1],points_proj_idx[row,0]]:
                # Shift based on ROI
                points_proj[row,0] -= roi[1]
                points_proj[row,1] -= roi[0]
                crop_ply_data.append(np.append(points_proj[row,:], points[row,:]))

        if debug:
            ##############
            temp_show = np.zeros((720,1280,3),dtype=np.uint8)
            cv2.imshow("After",vizDepth_new(np.asarray(crop_ply_data),temp_show))
            cv2.waitKey(1)
            ##############

        # ply_viz = np.zeros((720,1280,3),dtype=np.uint8)
        # for row in range(len(crop_ply_data)):
        #     pix = rs.rs2_project_point_to_pixel(self.intrinsics, crop_ply_data[row][2:5])
        #     pix = [round(x) for x in pix] # round to ints

        #     z = round(crop_ply_data[row][4] * -100)
        #     ply_viz[pix[1],pix[0]] = (30,z,30)

        # cv2.imshow("PLY VIZ", ply_viz)
        # cv2.waitKey(0)


        """
        Get segmented image out
        """
        mask_img = np.zeros((mask.shape[0],mask.shape[1],3))
        for idx in range(3):
            mask_img[:,:,idx] = mask
        output_image = np.multiply(image, mask_img).astype(np.uint8)
        output_image = output_image[roi[0]:roi[2],roi[1]:roi[3]]

        if debug:
            #####################
            temp_show_crop = temp_show[:,0:800]
            temp_show_crop = output_image * .5 + temp_show_crop *.5
            cv2.imshow("output_overlay",temp_show_crop.astype(np.uint8))
            print("Cycle Complete")
            cv2.waitKey(0)
            ######################

        return output_image, crop_ply_data

