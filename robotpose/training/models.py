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
from typing import Iterable

import numpy as np

from ..CompactJSONEncoder import CompactJSONEncoder
from ..constants import NUM_MODELS_TO_KEEP
from ..paths import Paths as p

INFO_JSON = os.path.join(p().MODELS, 'models.json')

class ModelData():
    def __init__(self, input_dict = None, **kwargs):
        self.__dict__ = {'id':'', 'dataset':'', 'dataset_size': 0, 'train_size':0,
        'valid_size': 0, 'classes':[], 'epochs_trained': 0, 'date_trained':'','benchmarks':[]}
        if input_dict is not None:
            assert type(input_dict) in [str, dict]
            input_dict = input_dict if type(input_dict) is dict else self._read(input_dict)
            self.__dict__.update((k, input_dict[k]) for k in input_dict.keys() if k in self.__dict__.keys())
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in self.__dict__.keys())
        self.train_ratio = self.train_size / self.dataset_size

    def _read(self, filepath):
        filepath = filepath if filepath.endswith('ModelData.json') else os.path.join(filepath,'ModelData.json')
        with open(filepath,'r') as f:
            return json.load(f)

    def write(self, folder_path):
        path = os.path.join(folder_path,'ModelData.json')
        with open(path,'w') as f:
            f.write(CompactJSONEncoder(indent=4).encode(self.__dict__).replace('\\','/'))

    def __iter__(self) -> Iterable:
        return iter([[key, self.__dict__[key]] for key in self.__dict__.keys()])

    def __repr__(self) -> str:
        return str(self.__dict__)




class ModelInfo():
    def __init__(self):
        self._cleanup()
        self.update()
    
    def update(self):
        data_files = [os.path.join(r,x) for r,d,y in os.walk(p().MODELS) for x in y if x.endswith('ModelData.json')]
        not_info = {}
        self.info = {}

        for datafile in data_files:
            data = ModelData(datafile)
            data.epochs_trained = self._getEpochs(datafile)
            not_info[data.id] = dict(data)
            self.info[data.id] = data

        self.num_total = len(self.info)

        with open(INFO_JSON,'w') as f:
            f.write(CompactJSONEncoder(indent=4).encode(not_info).replace('\\','/'))

    def _getEpochs(self, datafile_path, cleanup = True):
        folder = datafile_path.replace('ModelData.json','')
        modelfiles = [x for x in os.listdir(folder) if x.endswith('.h5')]
        epoch = [int(x.split('.')[1].split('-')[0]) for x in modelfiles]

        while cleanup and (len(epoch) > NUM_MODELS_TO_KEEP):
            to_del = min(epoch)
            for modelfile in modelfiles:
                if int(modelfile.split('.')[1].split('-')[0]) == to_del:
                    os.remove(os.path.join(folder, modelfile))

            modelfiles = [x for x in os.listdir(folder) if x.endswith('.h5')]
            epoch = [int(x.split('.')[1].split('-')[0]) for x in modelfiles]

        epoch.append(0)

        return max(epoch)

    def _cleanup(self):
        data_files = [os.path.join(r,x) for r,d,y in os.walk(p().MODELS) for x in y if x.endswith('ModelData.json')]
        data_dirs = [x.replace('ModelData.json','') for x in data_files]

        for folder in data_dirs:
            if len(os.listdir(folder)) == 1:
                os.remove(os.path.join(folder,'ModelData.json'))
                os.rmdir(folder)


class ModelManager(ModelInfo):
    def __init__(self):
        super().__init__()

    def allocateNew(self, dataset, classes, name=None):
        if name is None:
            letters = string.ascii_uppercase
            pick = True
            while pick:
                name = ''.join(random.choice(letters) for i in range(4))
                if name not in self.info.keys(): pick = False
        folder_path = os.path.join(p().MODELS, name)
        os.mkdir(folder_path)

        from ..data import Dataset
        ds = Dataset(dataset)
        folder = ds.link_anno_path
        train_length = len(os.listdir(os.path.join(folder,'train'))) // 2
        valid_length = len(os.listdir(os.path.join(folder,'test'))) // 2

        md = ModelData(id = name,
            dataset = dataset, dataset_size = int(ds.length),
            train_size = train_length, valid_size = valid_length,
            classes = classes, date_trained = str(datetime.now()))
        md.write(folder_path)

        return folder_path

    def loadByID(self, id):
        assert id in self.info.keys(), f"id {id} not found"

        folder = os.path.join(p().MODELS,id)
        files = [file for file in os.listdir(folder) if file.endswith('.h5')]
        files.sort()
        
        return os.path.join(folder,files[-1])   # Always return best

    def dynamicLoad(self, kwarg_dict = None,**kwargs):
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
        
        static_kwargs = {'dataset','classes','benchmark'}
        dynamic_kwargs = {'dataset_size', 'train_size', 'valid_size','train_ratio',
            'valid_ratio','used_ratio','epochs_trained'}
        dynamic_above = {x+'_above' for x in dynamic_kwargs}
        dynamic_below = {x+'_below' for x in dynamic_kwargs}
        dynamic_kwargs.update(dynamic_above)
        dynamic_kwargs.update(dynamic_below)

        for key in kwargs.keys(): assert key in dynamic_kwargs.union(static_kwargs), f"Unknown kwarg '{key}'"

        def apply_kwargs(remaining):

            def get_min(remaining, key):
                min_val = min([getattr(x,key) for x in remaining.values()])
                return {k:v for k,v in remaining.items() if getattr(v,key) == min_val}

            def get_max(remaining, key):
                max_val = max([getattr(x,key) for x in remaining.values()])
                return {k:v for k,v in remaining.items() if getattr(v,key) == max_val}


            for key, value in kwargs.items():
                current_state = remaining.copy()
                if len(remaining) == 1: return remaining

                if key in static_kwargs:
                    if key == 'benchmark':
                        pass #TODO: Add in benchmark
                    else:
                        remaining = {k:v for k,v in remaining.items() if getattr(v,key) == value}
                        if len(remaining) == 0:
                            remaining = current_state
                            print(f"Not using {key}={value} for model selection; Not satisfied by any remaining models.")

                else:
                    if key in dynamic_above:
                        remaining = {k:v for k,v in remaining.items() if getattr(v,key) >= value}
                        if len(remaining) == 0:
                            remaining = current_state
                            print(f"{key}={value} not satisfied for model selection; Using maximum value instead.")
                            return get_max(remaining, key)

                    elif key in dynamic_below:
                        remaining = {k:v for k,v in remaining.items() if getattr(v,key) <= value}
                        if len(remaining) == 0:
                            remaining = current_state
                            print(f"{key}={value} not satisfied for model selection; Using minimum value instead.")
                            return get_min(remaining, key)

                    else:
                        if abs(value) != np.inf:
                            min_diff = min([abs(value - getattr(x,key)) for x in remaining.values()])
                            remaining = {k:v for k,v in remaining.items() if abs(value - getattr(v,key)) == min_diff}
                        else:
                            if value == np.inf:
                                return get_max(remaining, key)
                            else:
                                return get_min(remaining, key)

            return remaining

        remaining = self.info.copy()

        remaining = apply_kwargs(remaining)
        if len(remaining) > 1:
            print(f"\n{len(remaining)} models match the chosen selection. Choosing most recently trained.")

            deltas = [datetime.now() - datetime.strptime(x.date_trained,'%Y-%m-%d %H:%M:%S.%f') for x in remaining.values()]
            deltas_sec = [d.total_seconds() for d in deltas]
            min_idx = deltas_sec.index(min(deltas_sec))
            datas = [x for x in remaining.values()]
            id = datas[min_idx].id
        elif len(remaining) == 1:
            id = [x.id for x in remaining.values()][0]
        else:
            return None

        return self.loadByID(id)

