# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import json
from json.decoder import JSONDecodeError
import numpy as np
import os
import tempfile
import zipfile
import shutil

import h5py
from numpy.core.numeric import False_

from ..paths import Paths as p
from .building import Builder
from ..CompactJSONEncoder import CompactJSONEncoder


INFO_JSON = os.path.join(p().DATASETS, 'datasets.json')
DATASET_VERSION = 7.0
                

class DatasetInfo():
    def __init__(self):
        self._update()

    def get(self):
        count = 0
        while True:
            try:
                with open(INFO_JSON, 'r') as f:
                    self.data = json.load(f)
                break
            except JSONDecodeError:
                count += 1
            if count > 99999:
                raise JSONDecodeError
        return self.data

    def unique_sets(self):
        datasets = set()
        datasets.update(self.compiled_sets())
        datasets.update(self.data['uncompiled']['names'])

        datasets = list(datasets)
        datasets.sort()
        return datasets

    def compiled_sets(self):
        datasets = list(set(self.data['compiled']['names']))
        datasets.sort()
        return datasets
        

    def __str__(self):
        self.get()
        datasets = self.unique_sets()

        full = []
        raw = []
        for ds in datasets:
            full.append(ds in self.data['compiled']['names'])
            raw.append(ds in self.data['uncompiled']['names'])


        out = "\nAvailable Datasets:\n"
        for idx in range(len(datasets)):
            out += f"\t{datasets[idx]}:\t"
            for ls, tag in zip([full,raw],['Full','Raw']):
                if ls[idx]:
                    out += f"[{tag}] "
            out += '\n'

        return out

    def __repr__(self):
        return f"Dataset Information stored in {INFO_JSON}."

    def _update(self):
        uncompiled_paths = [ f.path for f in os.scandir(os.path.join(p().DATASETS,'raw')) if str(f.path).endswith('.zip') ]
        uncompiled_names = [ os.path.basename(os.path.normpath(x)).replace('.zip','') for x in uncompiled_paths ]

        compiled_paths, compiled_names = ([] for i in range(2))

        for dirpath, subdirs, files in os.walk(p().DATASETS):
            for file in files:

                if file.endswith('.h5'):
                    compiled_names.append(file.replace('.h5',''))
                    compiled_paths.append(os.path.join(dirpath, file))
                    
        info = {
            'compiled':{'names': compiled_names, 'paths': compiled_paths},
            'uncompiled':{'names': uncompiled_names, 'paths': uncompiled_paths}
        }
        while True:
            try:
                with open(INFO_JSON,'w') as f:
                    f.write(CompactJSONEncoder(indent=4).encode(info).replace('\\','/'))
                break
            except PermissionError:
                pass
            



class Dataset():
    """
    Class to access, build, and use data.
    """

    def __init__(
            self, 
            name,
            recompile = False,
            rebuild = False,
            permissions = 'r'
            ):

        self.permissions = permissions
        self.name = name


        info = DatasetInfo()

        d = info.get()

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

        if recompile and not building:
            self.exportCameraPose()
            self.recompile()
            self.load()
            self.importCameraPose()

        self.load()

        # Delete temp backup if there still
        if os.path.isfile(self.dataset_path.replace(".h5","_old.h5")): os.remove(self.dataset_path.replace(".h5","_old.h5"))


    def exportCameraPose(self):
        np.save(os.path.join(self.dataset_dir,'camera_pose.npy'), self.camera_pose)

    def importCameraPose(self):
        camera_pose = np.load(os.path.join(self.dataset_dir,'camera_pose.npy'))
        self.camera_pose[:] = camera_pose

    def load(self):
        file = h5py.File(self.dataset_path,self.permissions)
        self.attrs = dict(file.attrs)
        self.og_resolution = self.attrs['resolution']
        self.length = self.attrs['length']
        self.angles = file['angles']
        self.positions = file['positions']
        self.depthmaps = file['coordinates/depthmaps']
        self.og_img = file['images/original']
        self.seg_img = file['images/segmented']
        self.camera_pose = file['images/camera_poses']
        self.preview_img = file['images/preview']

        # Set paths
        self.body_anno_path = os.path.join(self.dataset_dir,'body_annotations')
        self.link_anno_path = os.path.join(self.dataset_dir,'link_annotations')
        self.og_vid_path = os.path.join(self.dataset_dir,'og_vid.avi')
        self.seg_vid_path = os.path.join(self.dataset_dir,'seg_vid.avi')

    def recompile(self):
        bob = Builder()
        bob.recompile(self.dataset_dir, DATASET_VERSION, self.name)

    def build_from_zip(self, zip_path):
        """
        Build dataset from a tempdir that is the extracted data
        """

        with tempfile.TemporaryDirectory() as tempdir:
            print("Extracting raw data...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tempdir)

            # Go in one folder if needed
            src_dir = tempdir
            if len(os.listdir(tempdir)) == 1:
                src_dir = os.path.join(tempdir,os.listdir(tempdir)[0])

            print("Attempting dataset build...\n\n")
            bob = Builder()
            return bob.build_full(src_dir, DATASET_VERSION, os.path.basename(os.path.normpath(zip_path)).replace('.zip',''))

    def __len__(self):
        if self.length is None:
            return 0
        else:
            return self.length  

    def __repr__(self):
        return f"RobotPose dataset located at {self.dataset_path}."

    def __str__(self):
        out = ''
        out += f"Name: {self.attrs['name']}\n"
        out += f"Length: {self.attrs['length']} Poses\n"
        out += f"Dataset Version: {self.attrs['version']}\n\n"
        out += f"Build Date: {self.attrs['build_date']}\n"
        out += f"Compile Date: {self.attrs['compile_date']}\n"
        out += f"Compile Time: {self.attrs['compile_time']}\n\n"
        out += f"Resolution: {self.attrs['resolution']}\n"
        out += f"Color Intrinsics: {self.attrs['color_intrinsics']}\n"
        out += f"Depth Intrinsics: {self.attrs['depth_intrinsics']}\n"
        out += f"Depth Scale: {self.attrs['depth_scale']}\n"
        return out




 