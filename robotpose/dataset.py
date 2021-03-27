# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import datetime
import json
import multiprocessing as mp
import numpy as np
import os
import time

import cv2
from deepposekit.io import initialize_dataset
import h5py
from tqdm import tqdm

from .multithread import crop
from . import paths as p
from .segmentation import RobotSegmenter
from .utils import workerCount


INFO_JSON = os.path.join(p.DATASETS, 'datasets.json')

DATASET_VERSION = 3.0
"""
Version 1.0: 3/7/2021
    Began versioning.
    Compatible versions should include the same integer base (eg 1.0 and 1.4).
    Backwards-Incompatiblity should be marked by a new integer base (eg going from 1.4 to 2.0).

Version 1.1: 3/7/2021
    Changed raw compilation from using folders to using zip files to save storage

Version 1.2: 3/8/2021
    Added position parsing

Version 1.3: 3/19/2021
    Added ability to not load images/ply data at all
    Added support for keypoint location information

Version 2.0: 3/20/2021
    Added crop data to facilitate keypoint annotation

Version 2.1: 3/24/2021
    Added multithreading to dataset compilation

Version 3.0: 3/24/2021
    Switched PLY data over to aligned image arrays

"""



def save_video(path, img_arr):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path,fourcc, 15, (img_arr.shape[2],img_arr.shape[1]))
    for img in img_arr:
        out.write(img)
    out.release()



def build(data_path):
    """
    Build dataset into usable format
    """

    build_start_time = time.time()

    name = os.path.basename(os.path.normpath(data_path))
    dest_path = os.path.join(p.DATASETS, name)

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

    # Parse JSONs
    json_path = [os.path.join(data_path, x) for x in jsons]
    ang_arr = np.zeros((length, 6), dtype=float)
    pos_arr = np.zeros((length, 6, 3), dtype=float)

    for idx, path in tqdm(zip(range(length), json_path), desc="Parsing JSON Joint Angles and Positions"):
        with open(path, 'r') as f:
            d = json.load(f)
        d = d['objects'][0]['joints']

        for sub_idx in range(6):
            ang_arr[idx,sub_idx] = d[sub_idx]['angle']
            pos_arr[idx,sub_idx] = d[sub_idx]['position']

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
    for idx, path in tqdm(zip(range(length), orig_img_path),total=length,desc="Parsing 2D Images"):
        orig_img_arr[idx] = cv2.imread(path)


    segmenter = RobotSegmenter()

    segmented_img_arr = np.zeros((length, segmenter.height(), segmenter.width(), 3), dtype=np.uint8)
    pointmap = np.zeros((length, segmenter.height(), segmenter.width(), 3), dtype=np.float64)
    mask_arr = np.zeros((length, img_height, img_width), dtype=bool)
    rois = np.zeros((length, 4))

    # Segment images
    for idx in tqdm(range(length),desc="Segmenting Images",colour='red'):
        mask_arr[idx], rois[idx] = segmenter.segmentImage(orig_img_arr[idx])
    rois = rois.astype(int)

    # Make iterable for pool
    ply_paths = [os.path.join(data_path,x) for x in plys] 
    crop_inputs = []
    for idx in range(length):
        crop_inputs.append((ply_paths[idx], orig_img_arr[idx], mask_arr[idx], rois[idx]))

    print("Running Crop Pool...")
    # Run pool to segment PLYs
    with mp.Pool(workerCount()) as pool:
        crop_outputs = pool.starmap(crop, crop_inputs)
    print("Pool Complete")

    for idx in tqdm(range(length),desc="Unpacking Pool Results"):
        pointmap[idx] = crop_outputs[idx][1]
        segmented_img_arr[idx] = crop_outputs[idx][0]
    

    # Write dataset
    dest_path = os.path.join(dest_path, name + '.h5')
    file = h5py.File(dest_path,'a')
    file.attrs['name'] = name
    file.attrs['version'] = DATASET_VERSION
    file.attrs['length'] = length
    file.attrs['build_date'] = str(datetime.datetime.now())
    file.attrs['compile_time'] = time.time() - build_start_time
    file.attrs['type'] = 'full'
    file.attrs['original_resolution'] = orig_img_arr[0].shape
    file.attrs['segmented_resolution'] = segmented_img_arr[0].shape
    file.create_dataset('angles', data = ang_arr)
    file.create_dataset('positions', data = pos_arr)
    coord_grop = file.create_group('coordinates')
    coord_grop.create_dataset('depthmaps', data = depthmap)
    coord_grop.create_dataset('pointmaps', data = pointmap)
    img_grp = file.create_group('images')
    img_grp.create_dataset('original', data = orig_img_arr)
    img_grp.create_dataset('segmented', data = segmented_img_arr)
    img_grp.create_dataset('rois', data = rois)

    # Save reference videos
    save_video(os.path.join(dest_path,"og_vid.avi"), orig_img_arr)
    save_video(os.path.join(dest_path,"seg_vid.avi"), segmented_img_arr)



def update_info():
    uncompiled_paths = [ f.path for f in os.scandir(os.path.join(p.DATASETS,'raw')) if str(f.path).endswith('.zip') ]
    uncompiled_names = [ os.path.basename(os.path.normpath(x)).replace('.zip','') for x in uncompiled_paths ]
    compiled_full_paths = []
    compiled_full_names = []
    compiled_train_paths = []
    compiled_train_names = []
    compiled_validate_paths = []
    compiled_validate_names = []
    compiled_test_paths = []
    compiled_test_names = []

    for dirpath, subdirs, files in os.walk(p.DATASETS):
        for file in files:

            def full():
                compiled_full_names.append(file.replace('.h5',''))
                compiled_full_paths.append(os.path.join(dirpath, file))
            def train():
                compiled_train_names.append(file.replace('.h5',''))
                compiled_train_paths.append(os.path.join(dirpath, file))
            def validate():
                compiled_validate_names.append(file.replace('.h5',''))
                compiled_validate_paths.append(os.path.join(dirpath, file))
            def test():
                compiled_test_names.append(file.replace('.h5',''))
                compiled_test_paths.append(os.path.join(dirpath, file))
            switch = {
                'full': full,
                'train': train,
                'validate': validate,
                'test': test
            }

            if file.endswith('.h5'):
                with h5py.File(os.path.join(dirpath, file),'r') as f:
                    if 'type' in f.attrs:
                        switch[f.attrs['type']]()
                

    info = {
        'compiled':{
            'full':{
                'names': compiled_full_names,
                'paths': compiled_full_paths
            },
            'train':{
                'names': compiled_train_names,
                'paths': compiled_train_paths
            },
            'validate':{
                'names': compiled_validate_names,
                'paths': compiled_validate_paths
            },
            'test':{
                'names': compiled_test_names,
                'paths': compiled_test_paths
            }
        },
        'uncompiled':{
            'names': uncompiled_names,
            'paths': uncompiled_paths
        }
    }

    with open(INFO_JSON,'w') as f:
        json.dump(info, f, indent=4)
                




class Dataset():
    def __init__(
            self, 
            name,
            type = 'full',
            skeleton = None,
            force_recompile = False
            ):

        compiled_datasets = [ f.path for f in os.scandir(p.DATASETS) if f.is_dir() and 'raw' not in str(f.path) and 'skeleton' not in str(f.path) ]
        compiled_names = [ os.path.basename(os.path.normpath(x)) for x in compiled_datasets ]

        uncompiled_datasets = [ f.path for f in os.scandir(os.path.join(p.DATASETS,'raw')) if str(f.path).endswith('.zip') ]
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
        self.seg_anno_path = os.path.join(self.path,'seg_anno')
        self.name = ds_name


        # Load dataset
        self.load(skeleton)







    def load(self, skeleton=None):
        print("\nLoading Dataset...")
        # Read into JSON to get dataset settings
        with open(os.path.join(self.path, 'ds.json'), 'r') as f:
            d = json.load(f)

        self.length = d['frames']
        

        # Read in og images
        if self.load_og:
            self.og_img = np.load(os.path.join(self.path, 'og_img.npy'))
            self.og_vid = cv2.VideoCapture(os.path.join(self.path, 'og_vid.avi'))

        # Read in seg images
        if self.load_seg:
            self.seg_img = np.load(os.path.join(self.path, 'seg_img.npy'))
            self.seg_vid = cv2.VideoCapture(os.path.join(self.path, 'seg_vid.avi'))
            self.crop_data = np.load(os.path.join(self.path, 'crop_data.npy'))
            

        # Read angles
        self.ang = np.load(os.path.join(self.path, 'ang.npy'))

        # Read positions
        self.pos = np.load(os.path.join(self.path, 'pos.npy'))

        # Read in point data
        if self.load_ply:
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
        
        print("Dataset Loaded.\n")


    def validate(self, path):
        # Check Versions

        return True


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
            dest_dir = os.path.join(p.DATASETS,os.path.basename(os.path.normpath(zip_path)).replace('.zip',''))

            # Build
            print("Attempting dataset build...\n\n")
            build(src_dir, dest_dir)

        return dest_dir






    def setSkeleton(self,skeleton_name):
        for file in [x for x in os.listdir(p.SKELETONS) if x.endswith('.csv')]:
            if skeleton_name in os.path.splitext(file)[0]:
                self.skeleton = os.path.splitext(file)[0]
                self.skeleton_path = os.path.join(p.SKELETONS, file)
                self.deepposeds_path = self.deepposeds_path.replace('.h5','_'+os.path.splitext(file)[0]+'.h5')

        for file in [x for x in os.listdir(p.SKELETONS) if x.endswith('.json')]:
            if self.skeleton in os.path.splitext(file)[0]:
                with open(os.path.join(p.SKELETONS, file),'r') as f:
                    self.keypoint_data = json.load(f)


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
        return f"RobotPose dataset of {self.length} frames. Using skeleton {self.skeleton}."
