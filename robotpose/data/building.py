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
import os
import time
from typing import List

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from ..constants import DEFAULT_CAMERA_POSE, THUMBNAIL_DS_FACTOR, VIDEO_FPS
from ..paths import Paths as p

"""Used for segmentation during compilation, which is now avoided whenever possible"""
#from ..training import ModelManager
#from .segmentation import RobotSegmenter 


def save_video(path: str, img_arr: np.ndarray):
    """Create a video from an image array"""
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path,fourcc, VIDEO_FPS, (img_arr.shape[2],img_arr.shape[1]))
    for img in img_arr:
        out.write(img)
    out.release()


class Builder():
    def __init__(self, compression_level: int = 2):
        """Create a builder.
        Usually named Bob.

        Parameters
        ----------
        compression_level : int, optional
            h5py compression level for large datasets, by default 2
        """
        self.compression_level, self.build_start_time = compression_level, time.time()

    def build_full(self, data_path: str, name: str = None) -> str:
        """Build a dataset from a data folder path

        Parameters
        ----------
        data_path : str
            Folder path
        name : str, optional
            Name to give dataset, by default will automatically generate

        Returns
        -------
        str
            Dataset .h5 file path
        """
        self._set_dest_path(data_path, name)
        self._get_filepaths_from_data_dir(data_path)
        self._load_json_data()
        self._load_imgs_and_depthmaps()
        self._make_preview()
       

        self._save_reference_video()
        self._make_camera_poses()
        return self._save_full()

    def remove_idxs(self, src: str, rm_idxs: List[int]):
        """Remove specified indicies from a dataset

        Parameters
        ----------
        src : str
            Source .h5 dataset file
        rm_idxs : List[int]
            Indicies to remove
        """
        self._read_full(src)
        self.dest_path = os.path.dirname(src)
        keep_idxs = np.array([x for x in range(self.length) if x not in rm_idxs])
        self._filter(keep_idxs)
        self._save_full()

    def build_subset(self, src: str, sub_type: str, idxs: List[int]):
        """Build a derivative dataset of an arbitrary 'type' extraced from a source dataset

        Parameters
        ----------
        src : str
            Source .h5 dataset file
        sub_type : str
            'Type' to give new dataset
        idxs : List[int]
            Indicies to extract
        """
        self._read_full(src)
        dst = src.replace('.h5',f'_{sub_type}.h5')
        self._write_subset(dst, sub_type, idxs)

    def build_subsets(self, src: List[str], sub_types: List[str], idxs: List[List[str]]):
        """Build multiple subsets at once. See build_subset() for details"""
        self._read_full(src)
        for tp, idx in zip(sub_types, idxs):
            dst = src.replace('.h5',f'_{tp}.h5')
            self._write_subset(dst, tp, np.array(idx))




    def _set_dest_path(self, data_path: str, name: str = None):
        """Set destination folder"""
        if name is None:
            name = os.path.basename(os.path.normpath(data_path))
        self.dest_path = os.path.join(p().DATASETS, name)
        self.name = name
        if not os.path.isdir(self.dest_path):
            os.mkdir(self.dest_path)

    def _get_filepaths_from_data_dir(self, data_path: str):
        """Find all data files and store both name and full path"""
        self.jsons_p, self.maps_p, self.imgs_p = [[os.path.join(r,x) for r,d,y in os.walk(data_path) for x in y if x.endswith(end)] for end in ['.json','.npy','.png']]
        self.jsons, self.maps, self.imgs = [[x.replace(data_path,'') for x in self.jsons_p] for y in [self.jsons_p, self.maps_p, self.imgs_p]]

        # Make sure overall dataset length is the same for each file type
        self.length = len(self.imgs)
        assert len(self.jsons) == len(self.maps) == self.length, "Unequal number of images, jsons, or maps"


    def _load_json_data(self):
        """Parse in a dataset's JSON info files"""
        self.ang_arr = np.zeros((self.length, 6), dtype=float)
        self.pos_arr = np.zeros((self.length, 6, 3), dtype=float)
        depth_scale, intrin_depth ,intrin_color = [set() for i in range(3)]

        # Parse JSONs
        for idx, path in tqdm(zip(range(self.length), self.jsons_p), total=self.length, desc="Parsing JSONs"):
            with open(path, 'r') as f:
                d = json.load(f)

            depth_scale.add(d['realsense_info'][0]['depth_scale'])
            intrin_depth.add(d['realsense_info'][0]['intrin_depth'])
            intrin_color.add(d['realsense_info'][0]['intrin_color'])

            d = d['objects'][0]['joints']

            for sub_idx in range(6):
                self.ang_arr[idx,sub_idx] = d[sub_idx]['angle']
                self.pos_arr[idx,sub_idx] = d[sub_idx]['position']

        assert len(depth_scale) == len(intrin_depth) ==  len(intrin_color) == 1,f'Camera settings must be uniform over the dataset.'

        self.depth_scale, self.intrin_depth, self.intrin_color = depth_scale.pop(), intrin_depth.pop(), intrin_color.pop()



    def _load_imgs_and_depthmaps(self):
        """Load in RGB and D images"""
        img = cv2.imread(self.imgs_p[0])
        self.img_height, self.img_width = img.shape[0:2]

        # Create image array
        self.orig_img_arr = np.zeros((self.length, self.img_height, self.img_width, 3), dtype=np.uint8)
        self.depthmap_arr = np.zeros((self.length, self.img_height, self.img_width), dtype=np.float64)

        for idx, path in tqdm(zip(range(self.length), self.imgs_p),total=self.length,desc="Parsing 2D Images"):
            self.orig_img_arr[idx] = cv2.imread(path)
        for idx, path in tqdm(zip(range(self.length), self.maps_p),total=self.length,desc="Parsing Depthmaps"):
            self.depthmap_arr[idx] = np.load(path)

        self.depthmap_arr *= self.depth_scale

    def _make_preview(self):
        """Make thumbnail images for the dataset"""
        self.thumbnails = np.zeros((self.length, self.img_height // THUMBNAIL_DS_FACTOR, self.img_width // THUMBNAIL_DS_FACTOR, 3), dtype=np.uint8)
        for idx in tqdm(range(self.length),desc="Creating Thumbnails"):
            self.thumbnails[idx] = cv2.resize(self.orig_img_arr[idx], (self.img_width // THUMBNAIL_DS_FACTOR, self.img_height // THUMBNAIL_DS_FACTOR))

    def _save_reference_video(self):
        """Save video for later reference"""
        save_video(os.path.join(self.dest_path,"og_vid.avi"), self.orig_img_arr)

    def _make_camera_poses(self):
        """Create an array of default camera poses"""
        self.camera_poses = np.vstack([DEFAULT_CAMERA_POSE] * self.length)

    def _save_full(self) -> str:
        """Save all raw data into a dataset

        Returns
        -------
        str
            Dest .h5 file data was saved to
        """
        dest = os.path.join(self.dest_path, self.name + '.h5')

        # Delete file if already present
        if os.path.isfile(dest): os.remove(dest)

        with tqdm(total=9, desc="Writing Dataset") as pbar:
            with h5py.File(dest,'a') as file:
                file.attrs['name'] = self.name
                file.attrs['length'] = self.length
                file.attrs['build_date'] = str(datetime.datetime.now())
                file.attrs['compile_date'] = str(datetime.datetime.now())
                file.attrs['compile_time'] = time.time() - self.build_start_time
                file.attrs['resolution'] = self.orig_img_arr[0].shape[:-1]
                file.attrs['depth_intrinsics'] = self.intrin_depth
                file.attrs['color_intrinsics'] = self.intrin_color
                file.attrs['depth_scale'] = self.depth_scale
                file.create_dataset('angles', data = self.ang_arr, compression="gzip",compression_opts=self.compression_level)
                pbar.update(1)
                file.create_dataset('positions', data = self.pos_arr, compression="gzip",compression_opts=self.compression_level)
                pbar.update(1)
                coord_grop = file.create_group('coordinates')
                dm = coord_grop.create_dataset('depthmaps', data = self.depthmap_arr, compression="gzip",compression_opts=self.compression_level)
                pbar.update(1)
                dm.attrs['depth_scale'] = self.depth_scale
                img_grp = file.create_group('images')
                img_grp.create_dataset('original', data = self.orig_img_arr, compression="gzip",compression_opts=self.compression_level)
                pbar.update(1)
                img_grp.create_dataset('preview', data = self.thumbnails)
                pbar.update(1)
                img_grp.create_dataset('camera_poses', data = self.camera_poses)
                pbar.update(1)
                path_grp = file.create_group('paths')
                path_grp.create_dataset('jsons', data = np.array(self.jsons, dtype=h5py.string_dtype()), compression="gzip",compression_opts=self.compression_level)
                pbar.update(1)
                path_grp.create_dataset('depthmaps', data = np.array(self.maps, dtype=h5py.string_dtype()), compression="gzip",compression_opts=self.compression_level)
                pbar.update(1)
                path_grp.create_dataset('images', data = np.array(self.imgs, dtype=h5py.string_dtype()), compression="gzip",compression_opts=self.compression_level)
                pbar.update(1)

        return dest

    def _read_full(self, path:str):
        """Read in all data from a dataset

        Parameters
        ----------
        path : str
            .h5 dataset file path
        """
        with tqdm(total=9, desc="Reading Full Dataset") as pbar:
            with h5py.File(path,'r') as file:
                self.attrs = dict(file.attrs)
                self.name = file.attrs['name']
                self.length = file.attrs['length']

                self.intrin_depth = file.attrs['depth_intrinsics']
                self.intrin_color = file.attrs['color_intrinsics']
                self.depth_scale = file.attrs['depth_scale']
                self.ang_arr = np.copy(file['angles'])
                pbar.update(1)
                self.pos_arr = np.copy(file['positions'])
                pbar.update(1)
                self.depthmap_arr = np.copy(file['coordinates/depthmaps'])
                pbar.update(1)

                self.orig_img_arr = np.copy(file['images/original'])
                pbar.update(1)
                self.thumbnails = np.copy(file['images/preview'])
                pbar.update(1)
                self.camera_poses = np.copy(file['images/camera_poses'])
                pbar.update(1)

                self.jsons = np.copy(file['paths/jsons'])
                pbar.update(1)
                self.maps = np.copy(file['paths/depthmaps'])
                pbar.update(1)
                self.imgs = np.copy(file['paths/images'])
                pbar.update(1)


    def _filter(self, idxs: List[int]):
        """Return only a portion of the dataset data

        Parameters
        ----------
        idxs : List[int]
            Indicies to include
        """
        self.length = len(idxs)
        self.ang_arr = self.ang_arr[idxs]
        self.pos_arr = self.pos_arr[idxs]
        self.depthmap_arr = self.depthmap_arr[idxs]
        self.orig_img_arr = self.orig_img_arr[idxs]
        self.thumbnails = self.thumbnails[idxs]
        self.camera_poses = self.camera_poses[idxs]
        self.jsons = self.jsons[idxs]
        self.maps = self.maps[idxs]
        self.imgs = self.imgs[idxs]


