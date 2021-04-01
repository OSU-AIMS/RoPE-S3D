# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import multiprocessing as mp
from robotpose.utils import workerCount
import numpy as np
import os
import tempfile

import cv2
import h5py
from labelme.label_file import LabelFile
from tqdm import tqdm

from .dataset import Dataset
from .render import Renderer
from . import paths as p
from robotpose import render

def makeMask(image):
    mask = np.zeros(image.shape[0:2], dtype=np.uint8)
    mask[np.where(np.all(image != (0,0,0), axis=-1))] = 255
    return mask


def maskImg(image):
    mask = makeMask(image)
    mask_ = np.zeros(image.shape, bool)
    for idx in range(image.shape[2]):
        mask_[:,:,idx] = mask
    mask_img = np.ones(image.shape, np.uint8) * 255
    return mask_img


def makeContours(image):
    thresh = makeMask(image)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def contourImg(image):
    contours = makeContours(image)
    img = np.copy(image)
    cv2.drawContours(img, contours, -1, (0,255,0), 1)
    return img



class SegmentationAnnotator():
    """
    Creates labelme-compatible annotation jsons and pngs for renders.
    """

    def __init__(self, color_dict = None):
        if color_dict is not None:
            self.color_dict = color_dict

    def setDict(self, color_dict):
        """ Set color dict if not specified in init"""
        self.color_dict = color_dict

    def annotate(self, image, render, path):
        """
        Annotates an image given a rendering

        Args:
            image (ndarray):
                The image to be annotated.
            render (ndarray):
                The generated rendering for the image.
            path (str):
                The location to save the annotation to.
                Do not include file extensions.
        """

        f = LabelFile()

        with tempfile.TemporaryDirectory() as tmpdir:
            cv2.imwrite(os.path.join(tmpdir,'img.png'), image)
            imageData = f.load_image_file(os.path.join(tmpdir,'img.png'))

        act_image_path = path + '.png'
        json_path = path
        if not json_path.endswith('.json'):
            json_path += '.json'


        shapes = []

        for label in self.color_dict:
            contours = self._get_contour(render, self.color_dict[label])

            for contour in contours:
                # Skip falses
                if len(contour) < 20:
                    continue
                # Convert to compatiable list
                contourlist = np.asarray(contour).tolist()
                contour_data = []
                for point in contourlist:
                    contour_data.append(point[0])
                
                # Make entry
                shape = {
                    "label": label,
                    "points": contour_data,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }

                # Add to shapes
                shapes.append(shape)

        # Save annotation file
        f.save(
            filename = json_path,
            shapes = shapes,
            imagePath = act_image_path,
            imageHeight = image.shape[0],
            imageWidth = image.shape[1],
            imageData = imageData
        )

        # Save image
        cv2.imwrite(act_image_path, image)



    def _mask_color(self,image, color):
        """ Return mask of where a certain color is"""
        mask = np.zeros(image.shape[0:2], dtype=np.uint8)
        mask[np.where(np.all(image == color, axis=-1))] = 255
        return mask

    def _get_contour(self,image, color):
        """ Return contour of a given color """
        contours, hierarchy = cv2.findContours(self._mask_color(image, color), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours


class AutomaticSegmentationAnnotator():
    def __init__(
            self,
            mesh_list,
            names,
            dataset,
            skeleton,
            mode = 'seg_full',
            mesh_path = p.ROBOT_CAD,
            mesh_type = '.obj',
            camera_pose = None,
            renderer = None
            ):

        modes = ['seg_full','seg']
        assert mode in modes, f"Mode must be one of: {modes}"

        if renderer is None:
            self.rend = Renderer(
                mesh_list,
                dataset,
                skeleton,
                name_list = names,
                mode = mode,
                mesh_path = mesh_path,
                mesh_type = mesh_type,
                camera_pose = camera_pose
                )
        else:
            self.rend = renderer
            self.rend.setMode(mode)

        color_dict = self.rend.getColorDict()
        self.anno = SegmentationAnnotator(color_dict)

        self.ds = Dataset(dataset, skeleton)

        if not os.path.isdir(self.ds.seg_anno_path):
            os.mkdir(self.ds.seg_anno_path)

        

    def run(self):

        color_imgs = []

        for frame in tqdm(range(self.ds.length),desc="Rendering Segmentation Masks"):
            self.rend.setPosesFromDS(frame)
            color,depth = self.rend.render()
            color_imgs.append(color)
            #self.anno.annotate(self.ds.img[frame],color,os.path.join(self.ds.seg_anno_path,f"{frame:05d}"))
            cv2.imshow("Automatic Segmentation Annotator", color)
            cv2.waitKey(1)

        cv2.destroyAllWindows()
        inputs = []

        for frame in range(self.ds.length):
            inputs.append((self.ds.img[frame],color_imgs[frame],os.path.join(self.ds.seg_anno_path,f"{frame:05d}")))

        print("Starting Segmentation Pool...")
        with mp.Pool(workerCount()) as pool:
            pool.starmap(self.anno.annotate, inputs)
        print("Pool Complete")



class KeypointAnnotator():

    def __init__(self, color_dict, dataset, skeleton):
        self.color_dict = color_dict
        self.ds = Dataset(dataset, skeleton)
        self.ds.makeDeepPoseDS()    # Make sure there is already a valid deeppose DS for the DS
        self.dpds = h5py.File(self.ds.deepposeds_path, 'r+')

    def setDict(self, color_dict):
        self.color_dict = color_dict

    def annotate(self, render, idx):
        anno = []
        for color in self.color_dict.values():
            anno.append(self._getColorMidpoint(render, color))

        anno = np.array(anno)
        anno[:,0] -= self.ds.rois[idx,1] # This used to be crop data, so it would be rois[idx, 1]

        self.dpds['annotated'][idx] = np.array([True]*len(self.color_dict))
        self.dpds['annotations'][idx] = anno

    
    def _getColorMidpoint(self, image, color):
        coords = np.where(np.all(image == color, axis=-1))
        avg_y = np.mean(coords[0])
        avg_x = np.mean(coords[1])
        return [avg_x, avg_y]



class AutomaticKeypointAnnotator():
    
    def __init__(
            self,
            mesh_list,
            names,
            dataset,
            skeleton,
            mesh_path = p.ROBOT_CAD,
            mesh_type = '.obj',
            camera_pose = None,
            renderer = None
            ):
        
        if renderer is None:
            self.rend = Renderer(
                mesh_list,
                dataset,
                skeleton,
                name_list = names,
                mode = 'key',
                mesh_path = mesh_path,
                mesh_type = mesh_type,
                camera_pose = camera_pose
                )
        else:
            self.rend = renderer
            self.rend.setMode('key')

        color_dict = self.rend.getColorDict()
        self.anno = KeypointAnnotator(color_dict,dataset,skeleton)

    def run(self):

        for frame in tqdm(range(self.anno.ds.length),desc="Annotating Keypoints"):
            self.rend.setPosesFromDS(frame)
            color,depth = self.rend.render()
            self.anno.annotate(color,frame)
            cv2.imshow("Automatic Keypoint Annotator", color)
            cv2.waitKey(1)

        cv2.destroyAllWindows()