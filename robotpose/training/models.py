# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import json
import os
import random
import string
from datetime import datetime
from typing import Iterable, List

import numpy as np
import PySimpleGUI as sg

from ..CompactJSONEncoder import CompactJSONEncoder
from ..constants import (MODEL_NAME_LENGTH, MODELDATA_FILE_NAME,
                         NUM_MODELS_TO_KEEP)
from ..paths import Paths as p
from ..utils import folder_size, folder_size_as_str, size_to_str


class ModelData():
    """Stores data about a single trained model"""

    def __init__(self, input_dict: dict = None, **kwargs):
        self.__dict__ = {'id':'', 'dataset':'', 'dataset_size': 0, 'train_size':0,
        'valid_size': 0, 'classes':[], 'epochs_trained': 0, 'date_trained':'','benchmarks':[]}

        if input_dict is not None:
            assert type(input_dict) in [str, dict]
            input_dict = input_dict if type(input_dict) is dict else self._read(input_dict) # Read dict in if a file path has been given

            # Add info in
            self.__dict__.update((k, input_dict[k]) for k in input_dict.keys() if k in self.__dict__.keys())
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__dict__.keys())

        # Set train ratio
        self.train_ratio = self.train_size / self.dataset_size

    def _read(self, filepath: str):
        """Read a dict and return data"""
        filepath = filepath if filepath.endswith(MODELDATA_FILE_NAME) else os.path.join(filepath,MODELDATA_FILE_NAME)
        with open(filepath,'r') as f:
            return json.load(f)

    def write(self, folder_path:str):
        path = os.path.join(folder_path,MODELDATA_FILE_NAME)
        with open(path,'w') as f:
            f.write(CompactJSONEncoder(indent=4).encode(self.__dict__).replace('\\','/'))

    def __iter__(self) -> Iterable:
        return iter([[key, self.__dict__[key]] for key in self.__dict__.keys()])

    def __repr__(self) -> str:
        return str(self.__dict__)

    def __getitem__(self, key):
        return self.__dict__[key]



class ModelInfo():
    """Stores data about all models that have been trained"""

    def __init__(self):
        self._cleanup()
        self.update()
    
    def update(self):
        """Get new data and write to new info file for reference"""
        data_files = [os.path.join(r,x) for r,d,y in os.walk(p().MODELS) for x in y if x.endswith(MODELDATA_FILE_NAME)]
        not_info = {}
        self.info = {}

        for datafile in data_files:
            # Update epochs for all
            data = ModelData(datafile)
            data.epochs_trained = self._getEpochs(datafile)

            # Store as dict and dict of modeldata
            not_info[data.id] = dict(data)
            self.info[data.id] = data

        self.num_total = len(self.info)

        # Write model info
        with open(p().MODEL_INFO_JSON,'w') as f:
            f.write(CompactJSONEncoder(indent=4).encode(not_info).replace('\\','/'))

    def _getEpochs(self, datafile_path: str, cleanup: bool = True) -> int:
        """Check each model for the maximum epoch listed in the checkpoints"""

        # Determine epochs based on file names
        folder = datafile_path.replace(MODELDATA_FILE_NAME,'')
        modelfiles = [x for x in os.listdir(folder) if x.endswith('.h5')]
        epoch = [int(x.split('.')[1].split('-')[0]) for x in modelfiles]

        # Reduce the number of kept checkpoints to that specified in the constants
        while cleanup and (len(epoch) > NUM_MODELS_TO_KEEP):
            to_del = min(epoch)
            for modelfile in modelfiles:
                if int(modelfile.split('.')[1].split('-')[0]) == to_del:
                    os.remove(os.path.join(folder, modelfile))

            modelfiles = [x for x in os.listdir(folder) if x.endswith('.h5')]
            epoch = [int(x.split('.')[1].split('-')[0]) for x in modelfiles]

        epoch.append(0) # Make sure list always has elements

        return max(epoch)

    def _cleanup(self):
        data_files = [os.path.join(r,x) for r,d,y in os.walk(p().MODELS) for x in y if x.endswith(MODELDATA_FILE_NAME)]
        data_dirs = [x.replace(MODELDATA_FILE_NAME,'') for x in data_files]

        # Check all model folders; if the only file in the folder is the modeldata file, delete the folder
        for folder in data_dirs:
            if len(os.listdir(folder)) == 1:
                os.remove(os.path.join(folder, MODELDATA_FILE_NAME))
                os.rmdir(folder)



class ModelManager(ModelInfo):
    """Allows for retrieval and management/storge of segmentation models"""

    def __init__(self):
        super().__init__()

    def allocateNew(self, dataset: str, classes: List[str], name: str =None) -> str:
        """Create a new location to store models

        Parameters
        ----------
        dataset : str
            [description]
        classes : List[str]
            Classes this model predicts
        name : str, optional
            Folder name/ID, by default is randomized

        Returns
        -------
        str
            Path that was allocated
        """
        # Create a name if none specified
        if name is None:
            # Pick a random ASCII string 
            letters = string.ascii_uppercase
            pick = True
            while pick:
                name = ''.join(random.choice(letters) for i in range(MODEL_NAME_LENGTH))
                if name not in self.info.keys(): pick = False
        folder_path = os.path.join(p().MODELS, name)
        os.mkdir(folder_path)

        # Get dataset train/valid length
        from ..data import Dataset
        ds = Dataset(dataset)
        folder = ds.link_anno_path
        train_length = len(os.listdir(os.path.join(folder,'train'))) // 2
        valid_length = len(os.listdir(os.path.join(folder,'test'))) // 2

        # Make a new modeldata file
        md = ModelData(id = name,
            dataset = dataset, dataset_size = int(ds.length),
            train_size = train_length, valid_size = valid_length,
            classes = classes, date_trained = str(datetime.now()))
        md.write(folder_path)

        return folder_path

    def loadByID(self, id: str) -> str:
        """Load the data from the most recently trained checkpoint of a model"""
        assert id in self.info.keys(), f"id {id} not found"

        # List checkpoints
        folder = os.path.join(p().MODELS,id)
        files = [file for file in os.listdir(folder) if file.endswith('.h5')]
        files.sort()
        
        return os.path.join(folder,files[-1])   # Always return best/last checkpoint

    def dynamicLoad(self, kwarg_dict: dict = None,**kwargs) -> str:
        """Choose the 'best' model that satisfies certain criteria.
        Criterion are applied in the order that they are given as kwargs.
        If not all critera can be met, they will be avoided or best matched.
        See kwargs section for valid criteria.

        Valid Kwargs
        ----------
        Static Kwargs:
            Must be met. No leniency in values. No attempt will be made to find the 'closest' match.

            dataset : str
                Dataset used to train model.
            classes : list(str)
                Classes output in model. Primarily useful in loading 'link' models.
            TODO: benchmark : str
                Benchmark name. Always returns best.
            

        Dynamic Kwargs:
            These kwargs can be specified alone to find the closest match.
            Use np.inf or -np.inf respectively to find greatest or least values
            e.g. train_size = 100 would find the model with training size closest to 100.

            These kwargs can also act as filters by appending either '_below' or '_above' to the name.
            e.g. train_size_above = 100 would attempt to only load models with a training size greater than 100.

            dataset_size / train_size / valid_size : int
                Number of poses in dataset used. Train/Valid are numbers used in training.
            train_ratio / valid_ratio / used_ratio : float
                Derivative quantities of numbers used to train.
            epochs_trained : int
                Number of epochs model was trained for.

        """
        self.update()
        if kwarg_dict is not None:
            kwargs.update(kwarg_dict)
        
        # Acceptable kwargs
        static_kwargs = {'dataset','classes','benchmark'}
        dynamic_kwargs = {'dataset_size', 'train_size', 'valid_size','train_ratio',
            'valid_ratio','used_ratio','epochs_trained'}

        # Add 'above' and 'below' filters
        dynamic_above = {x+'_above' for x in dynamic_kwargs}
        dynamic_below = {x+'_below' for x in dynamic_kwargs}
        dynamic_kwargs.update(dynamic_above)
        dynamic_kwargs.update(dynamic_below)

        # Notify if unknow kwarg
        for key in kwargs.keys(): assert key in dynamic_kwargs.union(static_kwargs), f"Unknown kwarg '{key}'"

        def apply_kwargs(remaining):

            # Get model with minimum of a key
            def get_min(remaining, key):
                min_val = min([getattr(x,key) for x in remaining.values()])
                return {k:v for k,v in remaining.items() if getattr(v,key) == min_val}

            # Get model with maximum of a key
            def get_max(remaining, key):
                max_val = max([getattr(x,key) for x in remaining.values()])
                return {k:v for k,v in remaining.items() if getattr(v,key) == max_val}


            for key, value in kwargs.items():
                current_state = remaining.copy()
                if len(remaining) == 1: return remaining

                if key in static_kwargs:
                    # Apply static kwargs
                    if key == 'benchmark':
                        pass #TODO: Add in benchmark
                    else:
                        remaining = {k:v for k,v in remaining.items() if getattr(v,key) == value}
                        if len(remaining) == 0:
                            remaining = current_state
                            print(f"Not using {key}={value} for model selection; Not satisfied by any remaining models.")

                else:
                    # Dynamic kwargs

                    if key in dynamic_above:    # 'Above' kwargs
                        remaining = {k:v for k,v in remaining.items() if getattr(v,key) >= value}
                        if len(remaining) == 0:
                            remaining = current_state
                            print(f"{key}={value} not satisfied for model selection; Using maximum value instead.")
                            return get_max(remaining, key)

                    elif key in dynamic_below:  # 'Below' kwargs
                        remaining = {k:v for k,v in remaining.items() if getattr(v,key) <= value}
                        if len(remaining) == 0:
                            remaining = current_state
                            print(f"{key}={value} not satisfied for model selection; Using minimum value instead.")
                            return get_min(remaining, key)

                    else:

                        # Equal Kwargs
                        if abs(value) != np.inf:
                            min_diff = min([abs(value - getattr(x,key)) for x in remaining.values()])
                            remaining = {k:v for k,v in remaining.items() if abs(value - getattr(v,key)) == min_diff}
                        else:
                            if value == np.inf:
                                return get_max(remaining, key)
                            else:
                                return get_min(remaining, key)

            return remaining

        # Apply filters
        remaining = self.info.copy()
        remaining = apply_kwargs(remaining)

        if len(remaining) > 1:
            # Multiple options
            print(f"\n{len(remaining)} models match the chosen selection. Choosing most recently trained.")

            # Find most recently trained model
            deltas = [datetime.now() - datetime.strptime(x.date_trained,'%Y-%m-%d %H:%M:%S.%f') for x in remaining.values()]
            deltas_sec = [d.total_seconds() for d in deltas]
            min_idx = deltas_sec.index(min(deltas_sec))
            datas = [x for x in remaining.values()]
            id = datas[min_idx].id

        elif len(remaining) == 1:

            # If only one, select it
            id = [x.id for x in remaining.values()][0]
        else:
            return None

        return self.loadByID(id)







class ModelTree(ModelInfo):
    """PySimpleGUI Tree of trained segmentation models
    
    Always uses the PySimpleGUI key of -model_tree-
    """

    def __init__(self):
        super().__init__()
        self.refresh()

    def _addDatasets(self):
        """Add each dataset as a section"""
        self.datasets = set()
        for dat in self.info.values():
            if dat['dataset'] not in self.datasets:
                size = 0
                for d in self.info.values():
                    if d['dataset'] == dat['dataset']:
                        size += folder_size(os.path.join(p().MODELS,d['id']))

                self.treedata.insert('',dat['dataset'],dat['dataset'],['','','',size_to_str(size)])
                self.datasets.add(dat['dataset'])


    def _addModelsToTree(self):
        """Add each individual model as a node"""
        for dat in self.info.values():
            time_string = datetime.strftime(datetime.strptime(dat['date_trained'],'%Y-%m-%d %H:%M:%S.%f'),'%m-%d %H:%M')
            self.treedata.insert(
                dat['dataset'],
                dat['id'],
                dat['id'],
                [time_string,dat['train_size'],dat['valid_size'],folder_size_as_str(os.path.join(p().MODELS,dat['id']))]
                )
            
    def refresh(self):
        """Recreate data when needed"""
        self.treedata = sg.TreeData()
        self._addDatasets()
        self._addModelsToTree()
        return self.treedata

    def __call__(self):
        """Return a PySimpleGUI Tree object with data"""
        return sg.Tree(self.treedata,('Trained On','Train #','Valid #','Size'), col_widths=[9,5,5,7], auto_size_columns=False, num_rows=8, key='-model_tree-')

    @property
    def data(self):
        """Return treedata, updating it first"""
        self.refresh()
        return self.treedata
