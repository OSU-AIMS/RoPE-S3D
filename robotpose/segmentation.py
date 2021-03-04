import pixellib
from pixellib.instance import custom_segmentation, instance_segmentation
import cv2
import os
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from .utils import makeIntrinsics
from tqdm import tqdm
from . import paths as p
import time


class RobotSegmenter():

    def __init__(self, resolution = (720,800), model_path = os.path.join(p.seg_models,'A.h5')):
        self.master = custom_segmentation()
        self.master.inferConfig(num_classes= 1, class_names= ["BG", "mh5"])
        self.master.load_model(model_path)
        self.crop_resolution = resolution
        self.intrinsics = makeIntrinsics()

    def segment(self, img, ply_path):
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
        look_up_dist = 25
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

        # Invert x Coords
        points[:,0] = points[:,0] * -1

        ply_data = np.zeros((points.shape[0], 3))
        ply_data[:,:] = points

        crop_ply_data = []

        """
        Ended up increasing exe time
        """
        # Determine XY search area of image
        # rs.rs2_deproject_pixel_to_point(intrin, (x,y)(px), depth (meters))
        # define_search_area = False
        # search_depth = 10
        # if define_search_area:
        #     # Take Xmin as roi[1] projected at midpoint of image
        #     # Take Xmax as roi[3] projected at midpoint of image
        #     X_search = (rs.rs2_deproject_pixel_to_point(self.intrinsics, (init_roi[1],image.shape[0]/2), search_depth),rs.rs2_deproject_pixel_to_point(self.intrinsics, (init_roi[3],image.shape[0]/2), search_depth))
        #     # Take Ymin as roi[0] projected at midpoint of image
        #     # Take Ymax as roi[2] projected at midpoint of image
        #     Y_search = (rs.rs2_deproject_pixel_to_point(self.intrinsics, (image.shape[1]/2,init_roi[0]), search_depth),rs.rs2_deproject_pixel_to_point(self.intrinsics, (image.shape[2]/2,init_roi[0]), search_depth))
        #     X_min = np.min(X_search)
        #     X_max = np.max(X_search)
        #     Y_min = np.min(Y_search)
        #     Y_max = np.max(Y_search)
        start_time = time.time()
        """
        """
        # Get pixel location of each point
        for row in range(points.shape[0]):
            # if define_search_area:
            #     if ply_data[row,0] < X_min or ply_data[row,0] > X_max or ply_data[row,1] < Y_min or ply_data[row,1] > Y_max:
            #         continue
            
            x,y = rs.rs2_project_point_to_pixel(self.intrinsics, ply_data[row,:])
            # If point is in mask, add to data
            if mask[round(y),round(x)]:
                crop_ply_data.append(np.append([x,y], ply_data[row,:]))

 
        """
        """
        # Store as numpy array
        crop_ply_data = np.asarray(crop_ply_data)

        print(f"{time.time()-start_time}")

        # ply_viz = np.zeros((720,1280,3),dtype=np.uint8)
        # for row in range(crop_ply_data.shape[0]):
        #     pix = rs.rs2_project_point_to_pixel(self.intrinsics, crop_ply_data[row,2:5])
        #     pix = [round(x) for x in pix] # round to ints

        #     z = round(crop_ply_data[row,4] * -100)
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
        #print(output_image.shape)
        #cv2.imshow("img",output_image)
        #cv2.waitKey(0)
        return output_image

