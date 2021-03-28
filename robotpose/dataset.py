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
from h5py._hl import dataset
import numpy as np
import os
import time

import cv2
from numpy.core.defchararray import join
from numpy.lib.function_base import append, select
from deepposekit.io import initialize_dataset
import h5py
from tqdm import tqdm

from .multithread import crop
from . import paths as p
from .segmentation import RobotSegmenter
from .utils import workerCount


INFO_JSON = os.path.join(p.DATASETS, 'datasets.json')
CONFIG_JSON = os.path.join(p.DATASETS, 'dataset_config.json')

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



def build_full(data_path, name = None):
    """
    Build dataset into usable format
    """
    build_start_time = time.time()

    if name is None:
        name = os.path.basename(os.path.normpath(data_path))
    dest_path = os.path.join(p.DATASETS, name)

    # Make dataset folder if it does not already exist
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)


    

    # Build lists of files
    # jsons = [x for x in os.listdir(data_path) if x.endswith('.json')]
    # maps = [x for x in os.listdir(data_path) if x.endswith('full.ply')]
    # imgs = [x for x in os.listdir(data_path) if x.endswith('og.png')]

    jsons = [os.path.join(d,x) for r,d,x in os.walk(data_path) if x.endswith('.json')]
    maps = [os.path.join(d,x) for r,d,x in os.walk(data_path) if x.endswith('.npy')]
    imgs = [os.path.join(d,x) for r,d,x in os.walk(data_path) if x.endswith('.png')]

    json_paths = [os.path.join(data_path, x) for x in jsons]
    map_paths = [os.path.join(data_path,x) for x in maps] 
    orig_img_paths = [os.path.join(data_path, x) for x in imgs]

    # Make sure overall dataset length is the same for each file type
    length = len(imgs)
    assert len(jsons) == len(maps) == length, "Unequal number of images, jsons, or maps"
    
    
    ang_arr = np.zeros((length, 6), dtype=float)
    pos_arr = np.zeros((length, 6, 3), dtype=float)
    depth_scale = set()
    intrin_depth = set()
    intrin_color = set()

    # Parse JSONs
    for idx, path in tqdm(zip(range(length), json_paths), desc="Parsing JSON Joint Angles and Positions"):
        with open(path, 'r') as f:
            d = json.load(f)
        depth_scale.add(d['realsense_info']['depth_scale'])
        intrin_depth.add(d['realsense_info']['intrin_depth'])
        intrin_color.add(d['realsense_info']['intrin_color'])

        d = d['objects'][0]['joints']

        for sub_idx in range(6):
            ang_arr[idx,sub_idx] = d[sub_idx]['angle']
            pos_arr[idx,sub_idx] = d[sub_idx]['position']

    assert len(depth_scale) == len(intrin_depth) ==  len(intrin_color) == 1,f'Camera settings must be uniform over the dataset.'

    depth_scale = depth_scale.pop()
    intrin_depth = intrin_depth.pop()
    intrin_color = intrin_color.pop()

    """
    Parse Images
    """

    # Get image dims
    img = cv2.imread(os.path.join(data_path,imgs[0]))
    img_height, img_width = img.shape


    # Create image array
    orig_img_arr = np.zeros((length, img_height, img_width, 3), dtype=np.uint8)
    depthmap_arr = np.zeros((length, img_height, img_width, 3), dtype=np.uint8)
    
    for idx, path in tqdm(zip(range(length), orig_img_paths),total=length,desc="Parsing 2D Images"):
        orig_img_arr[idx] = cv2.imread(path)
    for idx, path in tqdm(zip(range(length), map_paths),total=length,desc="Parsing Depthmaps"):
        depthmap_arr[idx] = np.load(path)

    depthmap_arr *= depth_scale

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
    crop_inputs = []
    for idx in range(length):
        crop_inputs.append((depthmap_arr[idx], orig_img_arr[idx], mask_arr[idx], rois[idx]))

    # Run pool to segment PLYs
    print("Running Crop Pool...")
    with mp.Pool(workerCount()) as pool:
        crop_outputs = pool.starmap(crop, crop_inputs)
    print("Pool Complete")

    for idx in tqdm(range(length),desc="Unpacking Pool Results"):
        segmented_img_arr[idx] = crop_outputs[idx][0]
        pointmap[idx] = crop_outputs[idx][1]
    

    # Write dataset
    dest_path = os.path.join(dest_path, name + '.h5')

    with h5py.File(dest_path,'a') as file:
        file.attrs['name'] = name
        file.attrs['version'] = DATASET_VERSION
        file.attrs['length'] = length
        file.attrs['build_date'] = str(datetime.datetime.now())
        file.attrs['compile_time'] = time.time() - build_start_time
        file.attrs['type'] = 'full'
        file.attrs['original_resolution'] = orig_img_arr[0].shape
        file.attrs['segmented_resolution'] = segmented_img_arr[0].shape
        file.attrs['depth_intrinsics'] = intrin_depth
        file.attrs['color_intrinsics'] = intrin_color
        file.attrs['depth_scale'] = depth_scale
        file.create_dataset('angles', data = ang_arr)
        file.create_dataset('positions', data = pos_arr)
        coord_grop = file.create_group('coordinates')
        dm = coord_grop.create_dataset('depthmaps', data = depthmap_arr)
        dm.attrs['depth_scale'] = depth_scale
        coord_grop.create_dataset('pointmaps', data = pointmap)
        img_grp = file.create_group('images')
        img_grp.create_dataset('original', data = orig_img_arr)
        img_grp.create_dataset('segmented', data = segmented_img_arr)
        img_grp.create_dataset('rois', data = rois)
        path_grp = file.create_group('paths')
        path_grp.create_dataset('jsons', data = np.array(jsons))
        path_grp.create_dataset('depthmaps', data = np.array(maps))
        path_grp.create_dataset('images', data = np.array(imgs))

    # Save reference videos
    save_video(os.path.join(dest_path,"og_vid.avi"), orig_img_arr)
    save_video(os.path.join(dest_path,"seg_vid.avi"), segmented_img_arr)




                

def get_config():
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



def stratified_dataset_split(joint_angles):
    """
    This is best used whenever positions are assumed to be uniformly distributed,
    as stratification will be more useful and representative.
    """
    min_angs = np.min(joint_angles, 0)
    max_angs = np.max(joint_angles, 0)
    joint_moves = min_angs != max_angs
    moving_joints = np.sum(joint_moves)

    config = get_config()
    valid_size = int(config['split_ratios']['validate'] * len(joint_angles))
    test_size = int(config['split_ratios']['test'] * len(joint_angles))
    train_size = len(joint_angles) - valid_size - test_size

    # Determine configurations
    def generate_zones(size):
        r = int(size ** (1 / moving_joints)) + 1
        cfg_mat = np.zeros((r ** moving_joints, moving_joints))
        
        for idx in range(moving_joints):
            cfg_mat[:,idx] = np.tile(np.repeat(np.arange(1,r+1), r ** idx), int((r ** (moving_joints))/r**(idx+1)))

        combos = np.prod(cfg_mat,1)
        diff = np.abs(combos - size)
        cfg_mat = cfg_mat[np.where(diff == np.min(diff))]

        if cfg_mat.shape[0] != 1:
            relative_weights = (max_angs - min_angs)[joint_moves]
            weighted = cfg_mat / np.tile(relative_weights, (cfg_mat.shape[0],1))
            scores = np.sum(weighted, 1)
            minima = np.argmin(scores)
            return cfg_mat[minima]
        else:
            return cfg_mat

    def sample(size, cfg_mat, used_indicies):
        zone_arr = np.copy(cfg_mat)
        zone_arr += 1
        intervals = []
        for idx in range(moving_joints):
            intervals.append(np.linspace(min_angs[idx], max_angs[idx], int(zone_arr[idx])))

        def make_sampling_arr():
            sampling_arr = joint_angles[:, joint_moves == True]
            np.delete(sampling_arr, used_indicies, 0)
            return sampling_arr

        idx_array = np.zeros((moving_joints,))

        def update_idx_arr():
            for idx in range(moving_joints):
                if idx_array[idx] == cfg_mat[idx]:
                    idx_array[idx] = 0
                    if idx == moving_joints - 1:
                        return False
                    else:
                        idx_array[idx + 1] += 1
                        return True

        def get_range_arr():
            range_arr = np.zeros((moving_joints,2))
            for idx in range(moving_joints):
                range_arr[idx,0] = intervals[idx][idx_array[idx]]
                range_arr[idx,1] = intervals[idx][idx_array[idx] + 1]
            return range_arr

        selected_idxs = []
        while(update_idx_arr()):
            range_arr = get_range_arr()
            sampling_arr = make_sampling_arr()
            for idx in range(moving_joints):
                sampling_arr[sampling_arr[:,idx] >= range_arr[idx,0],:]
                sampling_arr[sampling_arr[:,idx] <= range_arr[idx,1],:]

            if len(sampling_arr) != 0:
                c = np.random.choice(len(sampling_arr))
                choice = sampling_arr[c]
                actual = joint_angles[0]
                actual[joint_moves == True] = choice
                i = np.where((joint_angles == choice).all(axis=1))
                selected_idxs.append(i)
                used_indicies.append(i)

            idx_array[0] += 1

        # if len(selected_idxs) < size:
        #     unused = [x for x in range(joint_angles.shape[0]) if x not in used_indicies]
        #     extra = np.random.choice(unused, size - len(selected_idxs), replace=False)
        #     selected_idxs.extend(extra)
        #     used_indicies.extend(extra)
        
        return selected_idxs
       
    used = []
    test_idxs = sample(test_size, generate_zones(test_size), used)
    valid_idxs = sample(valid_size, generate_zones(valid_size), used)
    train_idxs = [x for x in range(joint_angles.shape[0]) if x not in used]

    valid_idxs.sort()
    test_idxs.sort()

    return train_idxs, valid_idxs, test_idxs



def simple_dataset_split(joint_angles):
    """
    This is best used whenever positions are assumed to be uniformly distributed,
    as stratification will be more useful and representative.
    """
    min_angs = np.min(joint_angles, 0)
    max_angs = np.max(joint_angles, 0)
    joint_moves = min_angs != max_angs
    moving_joints = np.sum(joint_moves)

    config = get_config()
    valid_size = int(config['split_ratios']['validate'] * len(joint_angles))
    test_size = int(config['split_ratios']['test'] * len(joint_angles))
    train_size = len(joint_angles) - valid_size - test_size

    def sample(size, used):
        selected = []
        unused = [x for x in range(joint_angles.shape[0]) if x not in used]
        intervals = np.linspace(min(unused),max(unused), int(len(unused)/size))
        for idx in range(intervals - 1):
            unused = [x for x in range(joint_angles.shape[0]) if x not in used]
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
                    compiled_test_names.append(file.replace('.h5','').repalce('_test',''))
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
            skeleton = None,
            ds_type = 'full',
            recompile = False,
            rebuild = False
            ):

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



        # Load dataset
        self.load(skeleton)

        if recompile:
            #self.recompile()
            pass


    



    def load(self, skeleton=None):
        print("\nLoading Dataset...")
        file = h5py.File(self.dataset_path,'r')
        self.attrs = dict(file.attrs)
        self.angles = file['angles']
        self.positions = file['positions']
        self.pointmaps = file['coordinates/pointmaps']
        self.og_img = file['images/original']
        self.seg_img = file['images/segmented']
        self.rois = file['images/rois']

        # Set deeppose dataset path
        self.deepposeds_path = os.path.join(self.dataset_dir,'deeppose.h5')
        self.seg_anno_path = os.path.join(self.dataset_dir,'seg_anno')

        # If a skeleton is set, change paths accordingly
        if skeleton is not None:
            self.setSkeleton(skeleton)

        print("Dataset Loaded.\n")


    def build(self,data_path):
        # Build dataset
        build_full(data_path)



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

            # Build
            print("Attempting dataset build...\n\n")
            build_full(src_dir, os.path.basename(os.path.normpath(zip_path)).replace('.zip',''))





    def _writeSubset(self,sub_type,idxs):

        subset_path = self.dataset_path.replace('.h5',f'_{sub_type}.h5')

        with h5py.File(subset_path,'a') as file:
            file.attrs = self.attrs

            file.attrs['length'] = len(idxs)
            file.attrs['build_date'] = str(datetime.datetime.now())
            file.attrs['compile_time'] = 0
            file.attrs['type'] = sub_type
            file.create_dataset('angles', data = self.angles[idxs])
            file.create_dataset('positions', data = self.positions[idxs])
            coord_grop = file.create_group('coordinates')
            coord_grop.create_dataset('pointmaps', data = self.pointmaps[idxs])
            img_grp = file.create_group('images')
            img_grp.create_dataset('original', data = self.og_img[idxs])
            img_grp.create_dataset('segmented', data = self.seg_img[idxs])
            img_grp.create_dataset('rois', data = self.rois[idxs])
            path_grp = file.create_group('paths')

            # These have to be copied directly from master
            with h5py.File(self.dataset_path,'r') as master:
                dm = coord_grop.create_dataset('depthmaps', data = master['coordinates/depthmaps'][idxs])
                dm.attrs['depth_scale'] = self.attrs['depth_scale']
                path_grp.create_dataset('jsons', data = master['paths/jsons'][idxs])
                path_grp.create_dataset('depthmaps', data = master['paths/depthmaps'][idxs])
                path_grp.create_dataset('images', data = master['paths/images'][idxs])



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
