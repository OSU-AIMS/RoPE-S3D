# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import json
import multiprocessing as mp
import os
import random
import shutil
import tempfile
from typing import List

import cv2
import numpy as np
from labelme.label_file import LabelFile
from tqdm import tqdm

from ..CompactJSONEncoder import CompactJSONEncoder
from ..simulation.render import DatasetRenderer
from ..utils import expandRegion, workerCount
from .dataset import Dataset
from ..paths import Paths as p


class Annotator():
    """
    Creates labelme-compatible annotation jsons and pngs for renders.
    """

    def __init__(self, pad_size: int = 5, color_dict: dict = None):
        """Creates labelme-compatible annotation jsons and pngs for renders.

        Parameters
        ----------
        pad_size : int, optional
            Pixel space to add around each joint, by default 5
        color_dict : dict, optional
            Joint color dct from renderer class. Can be specified with setDict(), by default None
        """
        self.pad_size = pad_size
        if color_dict is not None:
            self.color_dict = color_dict

    def setDict(self, color_dict: dict):
        """ Set color dict if not specified in init"""
        self.color_dict = color_dict

    def annotate(self, image: np.ndarray, render: np.ndarray, path: str):
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


    def _mask_color(self, image, color):
        """ Return mask of where a certain color is"""
        mask = np.zeros(image.shape[0:2], dtype=np.uint8)
        mask[np.where(np.all(image == color, axis=-1))] = 255
        mask = expandRegion(mask, self.pad_size)
        return mask

    def _get_contour(self, image: np.ndarray, color: List[int]):
        """ Return contour of a given color """
        contours, hierarchy = cv2.findContours(self._mask_color(image, color), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours


class AutomaticAnnotator():
    """
    Given an aligned dataset, produce the full-body or per-joint segmentation annotations.
    """
    def __init__(self, dataset: str,
            ds_renderer = None, preview: bool = True):
        """Create annotator for segmentation.

        Parameters
        ----------
        dataset : str
            Name of dataset to use.
        ds_renderer : robotpose.DatasetRenderer, optional
            Premade renderer class to use., by default None
        preview : bool, optional
            Whether or not to view renders before annotation., by default True
        """

        self.preview = preview

        if ds_renderer is None:
            self.rend = DatasetRenderer(dataset, 'seg')
        else:
            self.rend = ds_renderer
            self.rend.setMode('seg')

        self.anno = Annotator(color_dict = self.rend.color_dict, pad_size=3)
        self.ds = Dataset(dataset)

        self.dest_path = self.ds.link_anno_path
        if not os.path.isdir(self.dest_path): os.mkdir(self.dest_path)


    def run(self):

        with tqdm(total=100, desc="Annotating", position=0, colour='green') as pbar:
            color_imgs = []

            pbar.set_description("Rendering Segmentation Masks", refresh=True)

            # Render robot throughout entire dataset
            for frame in tqdm(range(self.ds.length),desc="Rendering",position=1,colour='red',leave=False):
                self.rend.setPosesFromDS(frame)
                color, depth = self.rend.render()
                color_imgs.append(color)
                if self.preview:
                    cv2.imshow("Automatic Segmentation Annotator", color)
                    cv2.waitKey(1)
                if ((frame + 1)*10)% self.ds.length == 0:
                    pbar.update(1)

            cv2.destroyAllWindows()
            pbar.set_description("Copying Image Array", refresh=True)
            
            # Copy the input image array to be allocated to parallel workers
            og_img = np.copy(self.ds.og_img)

            pbar.set_description("Packing Pool")
            pbar.update(19)

            # Clear out old annotations
            shutil.rmtree(self.dest_path)
            os.mkdir(self.dest_path)

            # Allocate data to workers
            inputs = []
            for frame in tqdm(range(self.ds.length),desc="Packing",position=1, leave=False, colour='red'):
                inputs.append((og_img[frame],color_imgs[frame],os.path.join(self.dest_path,f"{frame:05d}")))
            
            pbar.update(1)
            pbar.set_description("Running Pool", refresh=True)

            # Run parallel pool of annotators
            with mp.Pool(workerCount()) as pool:
                pool.starmap(self.anno.annotate, inputs)

            pbar.set_description("Organizing Data")
            pbar.update(59)

            # Split data into train, validate, and test sections
            splitter = Splitter(self.dest_path)

            #TODO: split correctly
            splitter.split(.4,.1)

            # Clean up progress bar
            pbar.set_description("Annotation")
            pbar.update(11)


class Splitter():
    def __init__(self, folder):
        self.folder = folder

        self.all, self.train, self.test, self.ignore = [[] for i in range(4)]
        self.past_split = True

        for fold in ['test', 'train', 'ignore']:
            if not os.path.isdir(os.path.join(self.folder,fold)):
                os.mkdir(os.path.join(self.folder,fold))

        self.load()

    def load(self):
        """
        Read in the annotations present in the current folder, reorganizing if needed
        """

        if os.path.isfile(os.path.join(self.folder,'split.json')):
            # Has been split before
            self.past_split = True

            with open(os.path.join(self.folder,'split.json'),'r') as f:
                split_data = json.load(f)

            def read_in(subfolder,validation):
                json = [x.replace('.json','') for x in os.listdir(os.path.join(self.folder,subfolder)) if x.endswith('.json')]
                png = [x.replace('.png','') for x in os.listdir(os.path.join(self.folder,subfolder)) if x.endswith('.png')]
                lst = [x for x in json if x in png]
                valid = [x in validation for x in lst]
                assert np.all(valid), f"Data error found for {subfolder} when loading data to split. Please re-annotate data."
                return lst

            self.train = read_in('train',split_data['train'])
            self.test = read_in('test',split_data['test'])
            self.ignore = read_in('ignore',split_data['ignore'])

        else:
            # New data
            self.past_split = False

            jsons_p = [os.path.join(r,x) for r,d,y in os.walk(self.folder) for x in y if x.endswith('.json') and x not in ['test.json','train.json']]
            png_p = [os.path.join(r,x) for r,d,y in os.walk(self.folder) for x in y if x.endswith('.png')]

            assert len(jsons_p) == len(png_p), "Error encountered in data split: unequal number of png's and json's"

            # Consolidate
            for file in [*jsons_p, *png_p]:
                shutil.move(file, os.path.join(self.folder, 'ignore', os.path.basename(file)))

            self.train = []
            self.test = []
            self.ignore = [x.replace('.json','') for x in os.listdir(os.path.join(self.folder,'ignore')) if x.endswith('.json')]

    
    def split(self, train_prop, valid_prop):
        """Given data and proportions, split into training, validation, and testing groups.

        This is conservative. Adding to a field will add files, keeping the same original files.

        Parameters
        ----------
        train_prop : float
            Proportion 0-1 of data to be allocated as training data
        valid_prop : float
            Proportion 0-1 of data to be allocated as validation data
        """
        tot = len(self.train) + len(self.test) + len(self.ignore)
        num_train = int(tot * train_prop)
        num_test = int(tot * valid_prop)

        # Move to ignore if too large
        for num, lst, name in zip((num_train,num_test),(self.train, self.test),('train','test')):
            if len(lst) > num:
                random.shuffle(lst)

                num_transfer = len(lst) - num

                for idx in range(num_transfer):
                    f = lst[idx]
                    self.ignore.append(f)
                    for e in ['.json','.png']:
                        shutil.move(os.path.join(self.folder, name, f"{f}{e}"),os.path.join(self.folder, 'ignore', f"{f}{e}"))

                del lst[:num_transfer]

        # Move into other lists
        for num, lst, name in zip((num_train,num_test),(self.train, self.test),('train','test')):
            if len(lst) < num:
                random.shuffle(self.ignore)

                num_transfer = num - len(lst)

                for idx in range(num_transfer):
                    f = self.ignore[idx]
                    lst.append(f)
                    for e in ['.json','.png']:
                        shutil.move(os.path.join(self.folder, 'ignore', f"{f}{e}"),os.path.join(self.folder, name, f"{f}{e}"))

                del self.ignore[:num_transfer]

        self.write()
            
    def write(self):
        """Write split.json data file"""
        with open(os.path.join(self.folder,'split.json'),'w') as f:
            f.write(CompactJSONEncoder(indent=4).encode({'train':self.train,'test':self.test,'ignore':self.ignore}))

    @property
    def ratios(self):
        """Current folder's proportions of train, validate, and testing data respectively"""
        tot = len(self.train) + len(self.test) + len(self.ignore)
        return len(self.train) / tot, len(self.test) / tot,len(self.ignore) / tot

    def ratios_equal(self, train_prop, valid_prop):
        """Returns bool of if, given the dataset size, the given ratios are reflected in the current split"""
        tot = len(self.train) + len(self.test) + len(self.ignore)
        num_train = int(tot * train_prop)
        num_test = int(tot * valid_prop)
        return num_train == len(self.train) and num_test == len(self.test)

    def resplit(self, train_prop, valid_prop):
        """Alias for split(), but skips checking anything if the requested proportions are already present"""
        if not self.ratios_equal(train_prop, valid_prop):
            self.split(train_prop, valid_prop)

def refresh_split(ds):
    # Read in config
    with open(p().SPLIT_CONFIG,'r') as f:
        data = json.load(f)
    data = data[ds]

    # Actually split if data is present
    expected_anno_dir = os.path.join(Dataset(ds).dataset_dir,"link_annotations")
    if os.path.isdir(expected_anno_dir):
        s = Splitter(expected_anno_dir)
        s.resplit(data['train'],data['validate'])
