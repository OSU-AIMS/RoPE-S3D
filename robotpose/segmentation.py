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


class RobotSegmenter():

    def __init__(self, resolution = (720,800), model_path = os.path.join(p.seg_models,'A.h5')):
        self.master = custom_segmentation()
        self.master.inferConfig(num_classes= 1, class_names= ["BG", "mh5"])
        self.master.load_model(model_path)
        self.crop_resolution = resolution
        self.intrinsics = makeIntrinsics(res = 1281)

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

        init_roi_width = roi[3] - roi[1]

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

        ply_data = np.zeros((points.shape[0], 5))
        ply_data[:,2:5] = points

        crop_ply_data = []

        # Get pixel location of each pointpoint
        for row in tqdm(range(points.shape[0])):
            pix = rs.rs2_project_point_to_pixel(self.intrinsics, ply_data[row,2:5])
            pix = [round(x) for x in pix] # round to ints

            # If point is in mask, add to data
            if mask[pix[1],pix[0]]:
                crop_ply_data.append(np.append(pix, ply_data[row, 2:5]))

        # Store as numpy array
        crop_ply_data = np.asarray(crop_ply_data)

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





if __name__ == "__main__":
    a = RobotSegmenter()
    a.segment(r'C:\Users\exley\OneDrive\Documents\GitHub\DeepPoseRobot\data\raw\set6_slu\2021030200001_og.png',r'C:\Users\exley\OneDrive\Documents\GitHub\DeepPoseRobot\data\raw\set6_slu\2021030200001_full.ply')

