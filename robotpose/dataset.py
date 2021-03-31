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



def save_video(path, img_arr):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path,fourcc, 15, (img_arr.shape[2],img_arr.shape[1]))
    for img in img_arr:
        out.write(img)
    out.release()


class Builder():
    def __init__(self):
        self.build_start_time = time.time()

    def build_full(self, data_path, name = None):
        self._set_dest_path(data_path, name)
        self._get_filepaths_from_data_dir(data_path)
        self._load_json_data()
        self._load_imgs_and_depthmaps()
        self._segment_images_and_maps()
        self._save_reference_videos()
        return self._save_full()

    def build_subset(self, src, sub_type, idxs):
        self._read_full(src)
        dst = src.replace('.h5',f'_{sub_type}.h5')
        self._write_subset(dst, sub_type, idxs)



    def _set_dest_path(self, data_path, name):
        if name is None:
            name = os.path.basename(os.path.normpath(data_path))
        self.dest_path = os.path.join(p.DATASETS, name)
        self.name = name
        if not os.path.isdir(self.dest_path):
            os.mkdir(self.dest_path)

    def _get_filepaths_from_data_dir(self, data_path):
        self.jsons_p = [os.path.join(r,x) for r,d,y in os.walk(data_path) for x in y if x.endswith('.json')]
        self.maps_p = [os.path.join(r,x) for r,d,y in os.walk(data_path) for x in y if x.endswith('.npy')]
        self.imgs_p = [os.path.join(r,x) for r,d,y in os.walk(data_path) for x in y if x.endswith('.png')]

        self.jsons = [x.replace(data_path,'') for x in self.jsons_p]
        self.maps = [x.replace(data_path,'') for x in self.maps_p]
        self.imgs = [x.replace(data_path,'') for x in self.imgs_p]

        # Make sure overall dataset length is the same for each file type
        self.length = len(self.imgs)
        assert len(self.jsons) == len(self.maps) == self.length, "Unequal number of images, jsons, or maps"


    def _load_json_data(self):
        self.ang_arr = np.zeros((self.length, 6), dtype=float)
        self.pos_arr = np.zeros((self.length, 6, 3), dtype=float)
        depth_scale = set()
        intrin_depth = set()
        intrin_color = set()

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

        self.depth_scale = depth_scale.pop()
        self.intrin_depth = intrin_depth.pop()
        self.intrin_color = intrin_color.pop()


    def _load_imgs_and_depthmaps(self):
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

    
    def _segment_images_and_maps(self):
        segmenter = RobotSegmenter()
        self.segmented_img_arr = np.zeros((self.length, segmenter.height(), segmenter.width(), 3), dtype=np.uint8)
        self.pointmap = np.zeros((self.length, segmenter.height(), segmenter.width(), 3), dtype=np.float64)
        self.mask_arr = np.zeros((self.length, self.img_height, self.img_width), dtype=bool)
        self.rois = np.zeros((self.length, 4))

        # Segment images
        for idx in tqdm(range(self.length),desc="Segmenting Images",colour='red'):
            self.mask_arr[idx], self.rois[idx] = segmenter.segmentImage(self.orig_img_arr[idx])
        self.rois = self.rois.astype(int)

        del segmenter

        # Make iterable for pool
        crop_inputs = []
        for idx in range(self.length):
            crop_inputs.append((self.depthmap_arr[idx], self.orig_img_arr[idx], self.mask_arr[idx], self.rois[idx]))

        # Run pool to segment PLYs
        print("Running Crop Pool...")
        with mp.Pool(workerCount()) as pool:
            crop_outputs = pool.starmap(crop, crop_inputs)
        print("Pool Complete")

        for idx in tqdm(range(self.length),desc="Unpacking Pool Results"):
            self.segmented_img_arr[idx] = crop_outputs[idx][0]
            self.pointmap[idx] = crop_outputs[idx][1]

    def _save_reference_videos(self):
        save_video(os.path.join(self.dest_path,"og_vid.avi"), self.orig_img_arr)
        save_video(os.path.join(self.dest_path,"seg_vid.avi"), self.segmented_img_arr)

    def _save_full(self):
        dest = os.path.join(self.dest_path, self.name + '.h5')

        with h5py.File(dest,'a') as file:
            file.attrs['name'] = self.name
            file.attrs['version'] = DATASET_VERSION
            file.attrs['length'] = self.length
            file.attrs['build_date'] = str(datetime.datetime.now())
            file.attrs['compile_date'] = str(datetime.datetime.now())
            file.attrs['compile_time'] = time.time() - self.build_start_time
            file.attrs['type'] = 'full'
            file.attrs['original_resolution'] = self.orig_img_arr[0].shape
            file.attrs['segmented_resolution'] = self.segmented_img_arr[0].shape
            file.attrs['depth_intrinsics'] = self.intrin_depth
            file.attrs['color_intrinsics'] = self.intrin_color
            file.attrs['depth_scale'] = self.depth_scale
            file.create_dataset('angles', data = self.ang_arr, compression="gzip")
            file.create_dataset('positions', data = self.pos_arr, compression="gzip")
            coord_grop = file.create_group('coordinates')
            dm = coord_grop.create_dataset('depthmaps', data = self.depthmap_arr, compression="gzip")
            dm.attrs['depth_scale'] = self.depth_scale
            coord_grop.create_dataset('pointmaps', data = self.pointmap, compression="gzip")
            img_grp = file.create_group('images')
            img_grp.create_dataset('original', data = self.orig_img_arr, compression="gzip")
            img_grp.create_dataset('segmented', data = self.segmented_img_arr, compression="gzip")
            img_grp.create_dataset('rois', data = self.rois, compression="gzip")
            path_grp = file.create_group('paths')
            path_grp.create_dataset('jsons', data = np.array(self.jsons, dtype=h5py.string_dtype()), compression="gzip")
            path_grp.create_dataset('depthmaps', data = np.array(self.maps, dtype=h5py.string_dtype()), compression="gzip")
            path_grp.create_dataset('images', data = np.array(self.imgs, dtype=h5py.string_dtype()), compression="gzip")

        return dest

    def _read_full(self, path):
        with h5py.File(path,'r') as file:
            self.attrs = file.attrs
            self.name = file.attrs['name']
            self.length = file.attrs['length']

            self.intrin_depth = file.attrs['depth_intrinsics']
            self.intrin_color = file.attrs['color_intrinsics']
            self.depth_scale = file.attrs['depth_scale']
            self.ang_arr = file['angles']
            self.pos_arr = file['positions']
            self.depthmap_arr = file['coordinates/depthmaps']
            self.pointmap = file['coordinates/pointmaps']

            self.orig_img_arr = file['images/original']
            self.segmented_img_arr = file['images/segmented']
            self.rois = file['images/rois']

            self.jsons = file['paths/jsons']
            self.maps = file['paths/depthmaps']
            self.imgs = file['paths/images']

    def _write_subset(self,path,sub_type,idxs):
        """Create a derivative dataset from a full dataset, using a subset of the data."""

        with h5py.File(path,'a') as file:
            file.attrs = self.attrs
            file.attrs['length'] = len(idxs)
            file.attrs['compile_date'] = str(datetime.datetime.now())
            file.attrs['compile_time'] = 0
            file.attrs['type'] = sub_type
            file.create_dataset('angles', data = self.ang_arr[idxs], compression="gzip")
            file.create_dataset('positions', data = self.pos_arr[idxs], compression="gzip")
            coord_grop = file.create_group('coordinates')
            dm = coord_grop.create_dataset('depthmaps', data = self.depthmap_arr[idxs], compression="gzip")
            dm.attrs['depth_scale'] = self.depth_scale
            coord_grop.create_dataset('pointmaps', data = self.pointmap[idxs], compression="gzip")
            img_grp = file.create_group('images')
            img_grp.create_dataset('original', data = self.orig_img_arr[idxs], compression="gzip")
            img_grp.create_dataset('segmented', data = self.segmented_img_arr[idxs], compression="gzip")
            img_grp.create_dataset('rois', data = self.rois[idxs], compression="gzip")
            path_grp = file.create_group('paths')
            path_grp.create_dataset('jsons', data = np.array(self.jsons[idxs], dtype=h5py.string_dtype()), compression="gzip")
            path_grp.create_dataset('depthmaps', data = np.array(self.maps[idxs], dtype=h5py.string_dtype()), compression="gzip")
            path_grp.create_dataset('images', data = np.array(self.imgs[idxs],dtype=h5py.string_dtype()), compression="gzip")

        
    def weld(self, path_a, path_b, dst_dir, name):
        a = h5py.File(path_a,'r')
        b = h5py.File(path_b,'r')
        dst = h5py.File(path_a,'r')

        a_attrs = a.attrs
        b_attrs = b.attrs

        for attribute in ['version','original_resolution','segmented_resolution','depth_intrinsics','color_intrinsics','depth_scale']:
            assert a_attrs[attribute] == b_attrs[attribute], f"{attribute} must be equal to join datasets"

        a_len = a.attrs['length']
        b_len = b.attrs['length']
        self.length = a_len + b_len

        self.name = name
        self.dest_path = dst_dir

        self.ang_arr = np.vstack((a['angles'],b['angles']))
        self.pos_arr = np.vstack((a['positions'],b['positions']))
        self.depthmap_arr = np.vstack((a['coordinates/depthmaps'],b['coordinates/depthmaps']))
        self.pointmap = np.vstack((a['coordinates/pointmaps'],b['coordinates/pointmaps']))
        self.orig_img_arr = np.vstack((a['images/original'],b['images/original']))
        self.segmented_img_arr = np.vstack((a['images/segmented'],b['images/segmented']))
        self.rois = np.vstack((a['images/rois'],b['images/rois']))
        self.jsons = np.vstack((a['paths/jsons'],b['paths/jsons']))
        self.maps = np.vstack((a['paths/depthmaps'],b['paths/depthmaps']))
        self.imgs = np.vstack((a['paths/images'],b['paths/images']))

        self._save_full()

                

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
            rebuild = False
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



    def load(self, skeleton=None):
        print("\nLoading Dataset...")
        file = h5py.File(self.dataset_path,'r')
        self.attrs = dict(file.attrs)
        self.length = self.attrs['length']
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


    def recompile(self):
        pass



    def build(self,data_path):
        bob = Builder()
        bob.build_full(data_path)


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
            return bob.build_full(src_dir, os.path.basename(os.path.normpath(zip_path)).replace('.zip',''))


    def writeSubset(self, sub_type, idxs):
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
        return f"RobotPose dataset located at {self.dataset_path}."

    def __str__(self):
        return str(self.attrs)
