# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import logging
import os
import shutil
import tempfile
import zipfile
from typing import List

import h5py
import numpy as np

from ..CompactJSONEncoder import CompactJSONEncoder
from ..paths import Paths as p
from .building import Builder

DATASET_VERSION = 7.0   # TODO: Is this really needed anymore?


class DatasetInfo():
    """Returns information about the available datasets
    TODO: Remove JSON File

    """
    def __init__(self):
        self._update()

    def _update(self):
        """Recreate the info dictionary and file"""
        # Find all .zip files in the data/raw directory
        uncompiled_paths = [ f.path for f in os.scandir(p().RAW_DATA) if str(f.path).endswith('.zip') ]
        uncompiled_names = [ os.path.basename(os.path.normpath(x)).replace('.zip','') for x in uncompiled_paths ]

        # Find all .h5 dataset files in the data directory
        compiled_paths, compiled_names = ([] for i in range(2))
        for dirpath, subdirs, files in os.walk(p().DATASETS):
            for file in files:

                if file.endswith('.h5'):
                    compiled_names.append(file.replace('.h5',''))
                    compiled_paths.append(os.path.join(dirpath, file))
        
        # Create and write info dict
        self.info = {
            'compiled':{'names': compiled_names, 'paths': compiled_paths},
            'uncompiled':{'names': uncompiled_names, 'paths': uncompiled_paths}
        }
        while True:
            try:
                with open(p().DATASET_INFO_JSON,'w') as f:
                    f.write(CompactJSONEncoder(indent=4).encode(self.info).replace('\\','/'))
                break
            except PermissionError:
                pass

    def __str__(self) -> str:
        datasets = self.unique_sets

        full = []
        raw = []
        for ds in datasets:
            full.append(ds in self.info['compiled']['names'])
            raw.append(ds in self.info['uncompiled']['names'])


        out = "\nAvailable Datasets:\n"
        for idx in range(len(datasets)):
            out += f"\t{datasets[idx]}:\t"
            for ls, tag in zip([full,raw],['Full','Raw']):
                if ls[idx]:
                    out += f"[{tag}] "
            out += '\n'

        return out

    def __repr__(self) -> str:
        return f"Dataset Information stored in {p().DATASET_INFO_JSON}."

    @property
    def unique_sets(self) -> List[str]:
        """All datasets, compiled or raw"""
        datasets = set()
        datasets.update(self.compiled_sets)
        datasets.update(self.info['uncompiled']['names'])

        datasets = list(datasets)
        datasets.sort()
        return datasets

    @property
    def compiled_sets(self) -> List[str]:
        """Datasets that have been compiled"""
        datasets = list(set(self.info['compiled']['names']))
        datasets.sort()
        return datasets if len(datasets) > 0 else [None]

            



class Dataset():
    """
    Class to access, build, and use data.
    """

    def __init__(
            self, 
            name: str,
            rebuild: bool = False,
            permissions: str = 'r'
            ):
        """Create or use a dataset

        Parameters
        ----------
        name : str
            Dataset name. corresponds to inital .zip file name and /data/ folder
        rebuild : bool, optional
            Entirely recreate the dataset from the .zip source file. Done automatically if needed, by default False
        permissions : str, optional
            h5py permissions to use to open the dataset file. Use 'a' to modify any data in the dataset, by default 'r'
        """

        self.permissions, self.name = permissions, name

        info = DatasetInfo()
        d = info.info

        if name in d['compiled']['names']:
            self.dataset_path = d['compiled']['paths'][d['compiled']['names'].index(name)]
            self.dataset_dir = os.path.dirname(self.dataset_path)
        
        building = False
        if name not in d['compiled']['names'] or rebuild:
            building = True
            # Not here, rebuild
            available = d['uncompiled']['names']
            matches = [name in x for x in available]
            if np.sum(matches) == 0:
                raise ValueError(f"The requested dataset is not available\n{info}")
            elif np.sum(matches) > 1:
                raise ValueError(f"The requested dataset name is ambiguous\n{info}")
            else:
                camera_pose_conserved = False
                if name in d['compiled']['names']:

                    # Can't just export pose using burner dataset; error if new attributes added
                    with h5py.File(self.dataset_path,'r') as f:
                        if "images/camera_poses" in f:
                            np.save(os.path.join(self.dataset_dir,'camera_pose.npy'), np.array(f['images/camera_poses']))
                            camera_pose_conserved = True

                    # Save old file just in case there's an error in building (don't wanna lose any data from in there)
                    shutil.move(self.dataset_path,self.dataset_path.replace(".h5","_old.h5"),)
                    
                self.dataset_path = self.build_from_zip(d['uncompiled']['paths'][matches.index(True)])
                self.dataset_dir = os.path.dirname(self.dataset_path)
                
                if camera_pose_conserved:
                    self.load()
                    self.importCameraPose()

        self.load()

        # Delete temp backup if there still
        if os.path.isfile(self.dataset_path.replace(".h5","_old.h5")): os.remove(self.dataset_path.replace(".h5","_old.h5"))

    def load(self):
        """Load data from .h5 dataset file"""
        self.file = h5py.File(self.dataset_path,self.permissions)
        self.attrs = dict(self.file.attrs)
        self.og_resolution = self.attrs['resolution']
        self.length = self.attrs['length']
        self.angles = self.file['angles']
        self.positions = self.file['positions']
        self.depthmaps = self.file['coordinates/depthmaps']
        self.og_img = self.file['images/original']
        self.camera_pose = self.file['images/camera_poses']
        self.preview_img = self.file['images/preview']
        self.intrinsics = self.attrs['color_intrinsics']

        # Set paths
        self.link_anno_path = os.path.join(self.dataset_dir,'link_annotations')
        self.og_vid_path = os.path.join(self.dataset_dir,'og_vid.avi')

    def build_from_zip(self, zip_path: str):
        """
        Build dataset from a tempdir containing data extracted from a zipped folder
        """

        with tempfile.TemporaryDirectory() as tempdir:
            logging.info("Extracting raw data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tempdir)

            # Go in one folder if needed
            src_dir = tempdir
            if len(os.listdir(tempdir)) == 1:
                src_dir = os.path.join(tempdir,os.listdir(tempdir)[0])

            logging.info("Attempting dataset build...\n\n")
            bob = Builder()
            return bob.build_full(src_dir, os.path.basename(os.path.normpath(zip_path)).replace('.zip',''))

    def exportCameraPose(self):
        """Save camera pose array as an npy file"""
        np.save(os.path.join(self.dataset_dir,'camera_pose.npy'), self.camera_pose)

    def importCameraPose(self):
        """Overwrite current camera pose array with the camera_pose.npy file contents"""
        camera_pose = np.load(os.path.join(self.dataset_dir,'camera_pose.npy'))
        self.camera_pose[:] = camera_pose


    def close_file(self):
        self.file.close()

    def __len__(self) -> int:
        if self.length is None:
            return 0
        else:
            return self.length  

    def __repr__(self) -> str:
        return f"RobotPose dataset located at {self.dataset_path}."

    def __str__(self) -> str:
        out = ''
        out += f"Name: {self.attrs['name']}\n"
        out += f"Length: {self.attrs['length']} Poses\n"
        out += f"Build Date: {self.attrs['build_date']}\n"
        out += f"Compile Date: {self.attrs['compile_date']}\n"
        out += f"Compile Time: {self.attrs['compile_time']}\n\n"
        out += f"Resolution: {self.attrs['resolution']}\n"
        out += f"Color Intrinsics: {self.attrs['color_intrinsics']}\n"
        out += f"Depth Intrinsics: {self.attrs['depth_intrinsics']}\n"
        out += f"Depth Scale: {self.attrs['depth_scale']}\n"
        return out




 