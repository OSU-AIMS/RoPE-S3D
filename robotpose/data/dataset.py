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

import h5py

from ..paths import Paths as p
from .building import Builder
from ..CompactJSONEncoder import CompactJSONEncoder


INFO_JSON = os.path.join(p().DATASETS, 'datasets.json')
CONFIG_JSON = os.path.join(p().DATASETS, 'dataset_config.json')

DATASET_VERSION = 6.0
                

def get_config():
    """Get dataset splits from config JSON"""

    if not os.path.isfile(CONFIG_JSON):
        config = {
            'split_ratios':{
                'train': .70,
                'validate': .20,
                'test': .10
            }
        }
        with open(CONFIG_JSON,'w') as f:
            json.dump(config, f, indent=4)

    with open(CONFIG_JSON, 'r') as f:
        d = json.load(f)

    assert np.round(np.sum(list(d['split_ratios'].values())),5) == 1, f"Dataset Splits Must Sum To 1"

    return d


def dataset_split(joint_angles):

    config = get_config()
    valid_size = int(config['split_ratios']['validate'] * len(joint_angles))
    test_size = int(config['split_ratios']['test'] * len(joint_angles))

    def sample(size, used):
        selected = []
        unused = [x for x in range(joint_angles.shape[0]) if x not in used]
        intervals = np.linspace(min(unused),max(unused), int(len(unused)/size))
        for idx in range(len(intervals) - 1):
            unused = np.array([x for x in range(joint_angles.shape[0]) if x not in used])
            pool = unused[unused >= intervals[idx]]
            pool = pool[pool <= intervals[idx + 1]]
            c = np.random.choice(pool)
            used.append(c)
            selected.append(c)

        if len(selected) < size:
            unused = [x for x in range(joint_angles.shape[0]) if x not in used]
            extra = np.random.choice(unused, size - len(selected), replace=False)
            selected.extend(extra)
            used.extend(extra)

        return selected

    used = []
    test_idxs = sample(test_size, used)
    valid_idxs = sample(valid_size, used)
    train_idxs = [x for x in range(joint_angles.shape[0]) if x not in used]

    valid_idxs.sort()
    test_idxs.sort()

    return train_idxs, valid_idxs, test_idxs





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
        datasets = set()
        for t in ['full','train','validate','test']:
            datasets.update(self.data['compiled'][t]['names'])

        datasets = list(datasets)
        datasets.sort()
        return datasets
        

    def __str__(self):
        self.get()
        datasets = self.unique_sets()

        full = []
        train = []
        validate = []
        test = []
        raw = []
        for ds in datasets:
            full.append(ds in self.data['compiled']['full']['names'])
            train.append(ds in self.data['compiled']['train']['names'])
            validate.append(ds in self.data['compiled']['validate']['names'])
            test.append(ds in self.data['compiled']['test']['names'])
            raw.append(ds in self.data['uncompiled']['names'])


        out = "\nAvailable Datasets:\n"
        for idx in range(len(datasets)):
            out += f"\t{datasets[idx]}:\t"
            for ls, tag in zip([full,train,validate,test,raw],['Full','Train','Validate','Test','Raw']):
                if ls[idx]:
                    out += f"[{tag}] "
            out += '\n'

        return out

    def __repr__(self):
        return f"Dataset Information stored in {INFO_JSON}."

    def _update(self):
        uncompiled_paths = [ f.path for f in os.scandir(os.path.join(p().DATASETS,'raw')) if str(f.path).endswith('.zip') ]
        uncompiled_names = [ os.path.basename(os.path.normpath(x)).replace('.zip','') for x in uncompiled_paths ]

        compiled_full_paths, compiled_full_names, compiled_train_paths, compiled_train_names, \
        compiled_validate_paths, compiled_validate_names, compiled_test_paths, compiled_test_names = ([] for i in range(8))

        for dirpath, subdirs, files in os.walk(p().DATASETS):
            for file in files:

                def full():
                    compiled_full_names.append(file.replace('.h5',''))
                    compiled_full_paths.append(os.path.join(dirpath, file))
                def train():
                    compiled_train_names.append(file.replace('.h5','').replace('_train',''))
                    compiled_train_paths.append(os.path.join(dirpath, file))
                def validate():
                    compiled_validate_names.append(file.replace('.h5','').replace('_validate',''))
                    compiled_validate_paths.append(os.path.join(dirpath, file))
                def test():
                    compiled_test_names.append(file.replace('.h5','').replace('_test',''))
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
                'full':{'names': compiled_full_names, 'paths': compiled_full_paths},
                'train':{'names': compiled_train_names, 'paths': compiled_train_paths},
                'validate':{'names': compiled_validate_names, 'paths': compiled_validate_paths},
                'test':{'names': compiled_test_names, 'paths': compiled_test_paths}
            },
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
            ds_type = 'full',
            recompile = False,
            rebuild = False,
            permissions = 'r',
            ):
        """
        Create a dataset instance, loading/building/compiling it if needed.

        Arguments:
        name: A string corresponding to the entire, or the start of, a dataset name.
        ds_type: The type of dataset to load. Full, train, validate, or test.
        recompile: Using the same raw files, generate intermediate data again.
        rebuild: Recreate entirely usng different allocations of files
        """
        self.permissions = permissions
        self.name = name

        valid_types = ['full', 'train', 'validate', 'test']
        assert ds_type in valid_types, f"Invalid Type. Must be one of: {valid_types}"

        info = DatasetInfo()

        d = info.get()
        
        if name in d['compiled'][ds_type]['names'] and not rebuild:
            building = False
            # Good job, it's here, load it
            self.type = ds_type
            self.dataset_path = d['compiled'][ds_type]['paths'][d['compiled'][ds_type]['names'].index(name)]
            self.dataset_dir = os.path.dirname(self.dataset_path)
        else:
            building = True
            # Not here, rebuild
            available = d['uncompiled']['names']
            matches = [name in x for x in available]
            if np.sum(matches) == 0:
                raise ValueError(f"The requested dataset is not available\n{info}")
            elif np.sum(matches) > 1:
                raise ValueError(f"The requested dataset name is ambiguous\n{info}")
            else:
                self.dataset_path = self.build_from_zip(d['uncompiled']['paths'][matches.index(True)])
                self.dataset_dir = os.path.dirname(self.dataset_path)

        if recompile and not building:
            self.recompile()

        self.load()


    def exportCameraPose(self):
        np.save(os.path.join(self.dataset_dir,'camera_pose.npy'), self.camera_pose)

    def importCameraPose(self):
        camera_pose = np.load(os.path.join(self.dataset_dir,'camera_pose.npy'))
        self.camera_pose[:] = camera_pose

    def makeNewSubsets(self):
        print("Writing Subsets...")
        idxs = dataset_split(self.angles)
        sub_types = ['train','validate','test']
        bob = Builder()
        bob.build_subsets(self.dataset_path, sub_types, idxs)

    def load(self):
        file = h5py.File(self.dataset_path,self.permissions)
        self.attrs = dict(file.attrs)
        self.og_resolution = self.attrs['resolution']
        self.length = self.attrs['length']
        self.angles = file['angles']
        self.positions = file['positions']
        self.pointmaps = file['coordinates/pointmaps']
        self.og_img = file['images/original']
        self.seg_img = file['images/segmented']
        self.camera_pose = file['images/camera_poses']

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
        out += f"Type: {self.attrs['type']}\n"
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




 