# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import json
import numpy as np
import os

from deepposekit.io import initialize_dataset
import h5py

from . import paths as p
from .building import Builder


INFO_JSON = os.path.join(p.DATASETS, 'datasets.json')
CONFIG_JSON = os.path.join(p.DATASETS, 'dataset_config.json')

DATASET_VERSION = 4.0
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

Version 4.0: 3/29/2021
    Switched to using .h5 format for datasets
"""
                

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
        with open(INFO_JSON, 'r') as f:
            self.data = json.load(f)
        return self.data

    def __str__(self):
        self.get()
        datasets = set()
        for t in ['full','train','validate','test']:
            datasets.update(self.data['compiled'][t]['names'])
        datasets.update(self.data['uncompiled']['names'])

        datasets = list(datasets)
        datasets.sort()

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
    """
    Class to access, build, and use data.
    """

    def __init__(
            self, 
            name,
            skeleton = None,
            ds_type = 'full',
            recompile = False,
            rebuild = False,
            permissions = 'r'
            ):
        """
        Create a dataset instance, loading/building/compiling it if needed.

        Arguments:
        name: A string corresponding to the entire, or the start of, a dataset name.
        skeleton: A string corresponding to the skeleton to load for the dataset.
        ds_type: The type of dataset to load. Full, train, validate, or test.
        recompile: Using the same raw files, generate intermediate data again.
        rebuild: Recreate entirely usng different allocations of files
        """
        self.permissions = permissions

        valid_types = ['full', 'train', 'validate', 'test']
        assert ds_type in valid_types, f"Invalid Type. Must be one of: {valid_types}"

        info = DatasetInfo()

        d = info.get()
        
        if name in d['compiled'][ds_type]['names'] and not rebuild:
            # Good job, it's here, load it
            self.type = ds_type
            self.dataset_path = d['compiled'][ds_type]['paths'][d['compiled'][ds_type]['names'].index(name)]
            self.dataset_dir = os.path.dirname(self.dataset_path)
        else:
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


        # Load dataset
        self.load(skeleton)

        if recompile:
            #self.recompile()
            pass    


    def makeNewSubsets(self):
        print("Writing Subsets...")
        idxs = dataset_split(self.angles)
        sub_types = ['train','validate','test']
        bob = Builder()
        bob.build_subsets(self.dataset_path, sub_types, idxs)


    def load(self, skeleton=None):
        print("\nLoading Dataset...")
        file = h5py.File(self.dataset_path,self.permissions)
        self.attrs = dict(file.attrs)
        self.og_resolution = self.attrs['original_resolution']
        self.seg_resolution = self.attrs['segmented_resolution']
        self.length = self.attrs['length']
        self.angles = file['angles']
        self.positions = file['positions']
        self.pointmaps = file['coordinates/pointmaps']
        self.og_img = file['images/original']
        self.seg_img = file['images/segmented']
        self.rois = file['images/rois']
        self.camera_pose = file['images/camera_poses']

        # Set paths
        self.deepposeds_path = os.path.join(self.dataset_dir,'deeppose.h5')
        self.seg_anno_path = os.path.join(self.dataset_dir,'seg_anno')
        self.og_vid_path = os.path.join(self.dataset_dir,'og_vid.avi')
        self.seg_vid_path = os.path.join(self.dataset_dir,'seg_vid.avi')

        # If a skeleton is set, change paths accordingly
        if skeleton is not None:
            self.setSkeleton(skeleton)

        print("Dataset Loaded.\n")


    def recompile(self):
        pass



    def build(self,data_path):
        bob = Builder()
        bob.build_full(data_path)
        if self.type == 'full':
            self.makeNewSubsets()


    def build_from_zip(self, zip_path):
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

            print("Attempting dataset build...\n\n")
            bob = Builder()
            return bob.build_full(src_dir, DATASET_VERSION, os.path.basename(os.path.normpath(zip_path)).replace('.zip',''))


    def _writeSubset(self, sub_type, idxs):
        bob = Builder()
        bob.build_subset(self.dataset_path, sub_type, idxs)


    def setSkeleton(self,skeleton_name):
        for file in [x for x in os.listdir(p.SKELETONS) if x.endswith('.csv')]:
            if skeleton_name in os.path.splitext(file)[0]:
                self.skeleton = os.path.splitext(file)[0]
                self.skeleton_path = os.path.join(p.SKELETONS, file)
                self.deepposeds_path = self.deepposeds_path.replace('.h5','_'+os.path.splitext(file)[0]+'.h5')

        for file in [x for x in os.listdir(p.SKELETONS) if x.endswith('.json')]:
            if self.skeleton in os.path.splitext(file)[0]:
                self.keypoint_data_path = os.path.join(p.SKELETONS, file)
                self.updateKeypointData()

    
    def updateKeypointData(self):
        with open(self.keypoint_data_path,'r') as f:
            self.keypoint_data = json.load(f)


    def makeDeepPoseDS(self, force=False):
        if force:
            initialize_dataset(
                images=np.array(self.seg_img),
                datapath=self.deepposeds_path,
                skeleton=self.skeleton_path,
                overwrite=True # This overwrites the existing datapath
            )
            return

        if not os.path.isfile(self.deepposeds_path):
            initialize_dataset(
                images=np.array(self.seg_img),
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
        return f"RobotPose dataset located at {self.dataset_path}."

    def __str__(self):
        return str(self.attrs)
