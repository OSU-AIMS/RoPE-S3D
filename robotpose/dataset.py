# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley


import os
import cv2
import numpy as np
from tqdm import tqdm
import json
import pyrealsense2 as rs
import open3d as o3d
import pickle
from robotpose import paths as p
from .segmentation import RobotSegmenter
import robotpose.utils as utils
import time
import datetime
from deepposekit.io import initialize_dataset


dataset_version = 1.2
"""
Version 1.0: 3/7/2021
    Began versioning.
    Compatible versions should include the same integer base (eg 1.0 and 1.4).
    Backwards-Incompatiblity should be marked by a new integer base (eg going from 1.4 to 2.0).

Version 1.1: 3/7/2022
    Changed raw compilation from using folders to using zip files to save storage

Version 1.2: 3/8/2022
    Added position parsing

"""



def build(data_path, dest_path = None):
    """
    Build dataset into usable format
    """

    build_start_time = time.time()

    if dest_path is None:
        name = os.path.basename(os.path.normpath(data_path))
        dest_path = os.path.join(p.datasets, name)

    # Make dataset folder if it does not already exist
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)

    # Build lists of files
    jsons = [x for x in os.listdir(data_path) if x.endswith('.json')]
    plys = [x for x in os.listdir(data_path) if x.endswith('full.ply')]
    imgs = [x for x in os.listdir(data_path) if x.endswith('og.png')]

    # Make sure overall dataset length is the same for each file type
    length = len(imgs)
    assert len(jsons) == len(plys) == length, "Unequal number of images, jsons, or plys"


    """
    Parse Images
    """

    # Get image dims
    img = cv2.imread(os.path.join(data_path,imgs[0]))
    img_height = img.shape[0]
    img_width = img.shape[1]

    # Create image array
    orig_img_arr = np.zeros((length, img_height, img_width, 3), dtype=np.uint8)

    # Get paths for each image
    orig_img_path = [os.path.join(data_path, x) for x in imgs]

    # Store images in array
    for idx, path in tqdm(zip(range(length), orig_img_path),desc="Parsing 2D Images"):
        orig_img_arr[idx] = cv2.imread(path)

    # Save array
    np.save(os.path.join(dest_path, 'og_img.npy'), orig_img_arr)

    # Save as a video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(dest_path,"og_vid.avi"),fourcc, 15, (img_width,img_height))
    for idx in range(length):
        out.write(orig_img_arr[idx])
    out.release()

    segmenter = RobotSegmenter()
    segmented_img_arr = np.zeros((length, segmenter.height(), segmenter.width(), 3), dtype=np.uint8)
    ply_data = []

    # Segment images and PLYS
    for idx in tqdm(range(length),desc="Segmenting"):
        ply_path = os.path.join(data_path,plys[idx])
        segmented_img_arr[idx,:,:,:], ply = segmenter.segmentImage(orig_img_arr[idx], ply_path)
        ply_data.append(ply)

    # Save segmented image array
    np.save(os.path.join(dest_path, 'seg_img.npy'), segmented_img_arr)

    # Save as a video (just for reference)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(os.path.join(dest_path,"seg_vid.avi"),fourcc, 15, (segmenter.width(),segmenter.height()))
    for idx in range(length):
        out.write(segmented_img_arr[idx])
    out.release()

    """
    Process PLY data
    """

    ply_data_nd = np.array(ply_data, dtype=object)
    np.save(os.path.join(dest_path,'ply.npy'),ply_data_nd)


    """
    Parse JSONs
    """
    json_path = [os.path.join(data_path, x) for x in jsons]
    ang_arr = np.zeros((length, 6), dtype=float)
    pos_arr = np.zeros((length, 6, 3), dtype=float)

    for idx, path in tqdm(zip(range(length), json_path), desc="Parsing JSON Joint Angles and Positions"):
        # Open file
        with open(path, 'r') as f:
            d = json.load(f)
        d = d['objects'][0]['joints']

        # Put data in array
        for sub_idx in range(6):
            ang_arr[idx,sub_idx] = d[sub_idx]['angle']
            pos_arr[idx,sub_idx] = d[sub_idx]['position']

    # Save JSON data as npy
    np.save(os.path.join(dest_path, 'ang.npy'), ang_arr)
    np.save(os.path.join(dest_path, 'pos.npy'), ang_arr)

    """
    Write dataset info file
    """
    # Make json info file
    info = {
        "ds_ver": dataset_version,
        "name": os.path.basename(os.path.normpath(dest_path)),
        "frames": length,
        "build_time": time.time() - build_start_time,
        "last_build": str(datetime.datetime.now())
    }

    with open(os.path.join(dest_path,'ds.json'),'w') as file:
        json.dump(info, file)





class Dataset():
    def __init__(self, name, skeleton=None, load_seg = True, load_og = False, primary = "seg"):
        
        self.load_seg = load_seg
        self.load_og = load_og

        compiled_datasets = [ f.path for f in os.scandir(p.datasets) if f.is_dir() and 'raw' not in str(f.path) and 'skeleton' not in str(f.path) ]
        compiled_names = [ os.path.basename(os.path.normpath(x)) for x in compiled_datasets ]

        uncompiled_datasets = [ f.path for f in os.scandir(os.path.join(p.datasets,'raw')) if str(f.path).endswith('.zip') ]
        uncompiled_names = [ os.path.basename(os.path.normpath(x)) for x in uncompiled_datasets ]

        compiled_matches = [x for x in compiled_names if x.startswith(name)]
        uncompiled_matches = [x for x in uncompiled_names if x.startswith(name)]

        if len(compiled_matches) == 0:
            # No compiled matches found, attempt to compile one
            if len(uncompiled_matches) == 0:
                # No matches found at all, list options
                raise ValueError(f"No matching datasets found. The following are availble:\n\tCompiled: {compiled_names}\n\tUncompiled: {[x.replace('.zip','') for x in uncompiled_names]}")
            if len(uncompiled_matches) > 1:
                # Multiple matches found
                raise ValueError(f"Multiple uncompiled sets were found with the given dataset name:\n\tGiven Name: {name}\n\tMatching Names: {uncompiled_matches}")
            else:
                # One uncompiled match found, compile
                ds_name = uncompiled_matches[0].replace('.zip','')
                print("No matching compiled datasets found.\nCompiling from raw data.\n\n")
                ds_path = self.compile_from_zip([x for x in uncompiled_datasets if uncompiled_matches[0] in x][0])
        elif len(compiled_matches) > 1:
            # Multiple matches found, raise error
            raise ValueError(f"Multiple compiled sets were found with the given dataset name:\n\tGiven Name: {name}\n\tMatching Names: {compiled_matches}")
        else:
            # One match found, set
            ds_name = compiled_matches[0]
            ds_path = [x for x in compiled_datasets if compiled_matches[0] in x][0]

        # There is now a dataset chosen, validate
        if self.validate(ds_path):
            # Validation sucessful, set paths
            self.path = ds_path
            self.name = os.path.basename(os.path.normpath(ds_path))
        else:
            # Attempt a recompile
            if len(uncompiled_matches) == 0:
                raise ValueError(f"The chosen dataset could not be validated and raw data for a rebuild cannot be found.")
            if len(uncompiled_matches) > 1:
                raise ValueError(f"The chosen dataset could not be validated and multiple raw data files for a rebuild are found.")
            else:
                # Actually recompile
                print("Dataset validation failed.\nAttempting recompile.\n\n")
                ds_name = uncompiled_matches[0].replace('.zip','')
                ds_path = self.compile_from_zip([x for x in uncompiled_datasets if uncompiled_matches[0] in x][0])

        
        self.path = ds_path
        self.name = ds_name


        # Load dataset
        self.load(skeleton)

        # Set paths, resolution
        if self.load_og:
            self.og_vid_path = os.path.join(self.path, 'og_vid.avi')
            self.resolution_og = self.og_img.shape[1:3]
            self.resolution = self.resolution_og
        if self.load_seg:
            self.seg_vid_path = os.path.join(self.path, 'seg_vid.avi')
            self.resolution_seg = self.seg_img.shape[1:3]
            self.resolution = self.resolution_seg


        # Set primary image and video types
        if self.load_seg and not self.load_og:
            primary = "seg"
        if self.load_og and not self.load_seg:
            primary = "og"

        if primary == "og":
            self.img = self.og_img
            self.vid = self.og_vid
            self.vid_path = self.og_vid_path
        else:
            self.img = self.seg_img
            self.vid = self.seg_vid
            self.vid_path = self.seg_vid_path
            if primary != "seg":
                print("Invalid primary media type selected.\nUsing seg.")



    def load(self, skeleton=None):
        print("\nLoading Dataset...")
        # Read into JSON to get dataset settings
        with open(os.path.join(self.path, 'ds.json'), 'r') as f:
            d = json.load(f)

        self.length = d['frames']

        # Read in og images
        if self.load_og:
            print("Reading orig images...")
            self.og_img = np.load(os.path.join(self.path, 'og_img.npy'))
            self.og_vid = cv2.VideoCapture(os.path.join(self.path, 'og_vid.avi'))

        # Read in seg images
        if self.load_seg:
            print("Reading segmented images...")
            self.seg_img = np.load(os.path.join(self.path, 'seg_img.npy'))
            self.seg_vid = cv2.VideoCapture(os.path.join(self.path, 'seg_vid.avi'))
            

        # Read angles
        print("Reading joint angles...")
        self.ang = np.load(os.path.join(self.path, 'ang.npy'))

        # Read angles
        print("Reading joint positions...")
        self.pos = np.load(os.path.join(self.path, 'pos.npy'))

        # Read in point data
        print("Reading 3D data...")
        ply_in = np.load(os.path.join(self.path, 'ply.npy'),allow_pickle=True)
        self.ply = []
        for entry in ply_in:
            entry = np.array(entry)
            self.ply.append(entry)

        # Set deeppose dataset path
        self.deepposeds_path = os.path.join(self.path,'deeppose.h5')

        # If a skeleton is set, change paths accordingly
        if skeleton is not None:
            self.setSkeleton(skeleton)
        
        print("Dataset Loaded.\n\n")


    def validate(self, path):
        """
        Check that all required elements of the dataset are present
        """
        ang = os.path.isfile(os.path.join(path,'ang.npy'))
        ds = os.path.isfile(os.path.join(path,'ds.json'))
        ply = os.path.isfile(os.path.join(path,'ply.npy'))
        seg_img = os.path.isfile(os.path.join(path,'seg_img.npy'))
        og_img = os.path.isfile(os.path.join(path,'og_img.npy'))
        seg_vid = os.path.isfile(os.path.join(path,'seg_vid.avi'))
        og_vid = os.path.isfile(os.path.join(path,'og_vid.avi'))
        
        if ds:
            with open(os.path.join(path, 'ds.json'), 'r') as f:
                d = json.load(f)

            try:
                if int(d['ds_ver']) != int(dataset_version):
                    print(f"Dataset Out of Date:\n\tDataset version:{d['ds_ver']}\n\tCurrent version: {dataset_version}")
                    return False
            except KeyError:
                print(f"Dataset Out of Date:\n\tDataset version: Unversioned\n\tCurrent version: {dataset_version}")
                return False
            

        return ang and ds and ply and seg_img and og_img and seg_vid and og_vid


    def build(self,data_path):
        # Build dataset
        build(data_path)



    def compile_from_zip(self, zip_path):
        """
        Build dataset from a tempdir that is the extracted data
        """
        import tempfile
        import zipfile
        with tempfile.TemporaryDirectory() as tempdir:
            print("Extracting raw data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tempdir)

            # Go in one folder if needed
            src_dir = tempdir
            if len(os.listdir(tempdir)) == 1:
                src_dir = os.path.join(tempdir,os.listdir(tempdir)[0])
            
            # Get final dataset path
            dest_dir = os.path.join(p.datasets,os.path.basename(os.path.normpath(zip_path)).replace('.zip',''))

            # Build
            print("Attempting dataset build...\n\n")
            build(src_dir, dest_dir)

        return dest_dir






    def setSkeleton(self,skeleton_name):
        for file in [x for x in os.listdir(p.skeletons) if x.endswith('.csv')]:
            if skeleton_name in os.path.splitext(file)[0]:
                self.skeleton = os.path.splitext(file)[0]
                self.skeleton_path = os.path.join(p.skeletons, file)
                self.deepposeds_path = self.deepposeds_path.replace('.h5','_'+os.path.splitext(file)[0]+'.h5')

    def makeDeepPoseDS(self, force=False):
        if force:
            initialize_dataset(
                images=self.seg_img,
                datapath=self.deepposeds_path,
                skeleton=self.skeleton_path,
                overwrite=True # This overwrites the existing datapath
            )
            return

        if not os.path.isfile(self.deepposeds_path):
            initialize_dataset(
                images=self.seg_img,
                datapath=self.deepposeds_path,
                skeleton=self.skeleton_path,
                overwrite=False # This overwrites the existing datapath
            )
        else:
            print("Using Precompiled Deeppose Dataset")


    def __len__(self):
        if self.length is None:
            return 0
        else:
            return self.length  

    def __repr__(self):
        return f"RobotPose dataset of {self.length} frames. Using skeleton {self.skeleton}"

    def og(self):
        return self.load_og

    def seg(self):
        return self.load_seg
