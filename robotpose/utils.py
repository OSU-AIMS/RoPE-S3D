# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import multiprocessing as mp
import os
import subprocess as sp
import time
from typing import Any, List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_gpu_memory() -> List[int]:
    """Query GPU's for amount of VRAM
    Modified from:
    https://stackoverflow.com/questions/59567226/how-to-programmatically-determine-available-gpu-memory-with-tensorflow

    Returns
    -------
    VRAM, List[int]
        Total GPU VRAM in bits for each GPU.
    """

    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
    memory_free_values = [int(x.split()[0])*8.389e6 for i, x in enumerate(memory_free_info)]
    return memory_free_values 


def workerCount() -> int:
    """Return number of threads to use as workers in multiprocessing"""
    cpu_count = mp.cpu_count()
    return int(min(cpu_count - 2, .75 * cpu_count))


def expandRegion(image, size, iterations = 1):
    kern = np.ones((size,size), dtype=np.uint8)
    return cv2.dilate(image, kern, iterations = iterations)


def str_to_arr(string: str) -> np.ndarray:
    """Convert a string of SLURBT to a (6,) numpy array of boolean values"""

    joints = ['S','L','U','R','B','T']
    out = np.zeros(6, bool)
    for letter in string.upper():
        out[joints.index(letter)] = True
    return out

def get_key(dict: dict, val: Any) -> Union[str, list]:
    """Return the keys of a certain dictionary value"""
    return list(dict.keys())[list(dict.values()).index(val)]


def reject_outliers_std(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def outlier_min_max(data, iqr_mult = 1.5):
    data = reject_outliers_iqr(data, iqr_mult)
    return np.min(data), np.max(data)

def reject_outliers_iqr(data, iqr_mult = 1.5):
    percentiles = np.percentile(data, [75, 25])
    iqr = np.subtract(*percentiles)
    max = percentiles[0] + iqr_mult * iqr
    min = percentiles[1] - iqr_mult * iqr
    data = data[data >= min]
    data = data[data <= max]
    return data



def get_extremes(mat: np.ndarray) -> List[int]:
    """Returns the limits of data in a boolean array

    Parameters
    ----------
    mat : ndarray
        Must be bool type

    Returns
    -------
    Extremes : list
        Min row, Max row, Min column, Max column
    """
    r, c = np.where(mat)
    return [min(r),max(r),min(c),max(c)]



def folder_size(path: str) -> int:
    """Return size of all files in folder in bytes"""
    size = 0
    for r, d, f in os.walk(path):
        for file in f:
            size += os.path.getsize(os.path.join(r, file))

    return size

def size_to_str(b: int) -> str:
    """Format a number of bytes as a string in B/KB/MB/GB"""
    postfixes = ['B','KB','MB','GB']
    vals = [b / (1000 ** p) for p in range(4)]
    v = min([x for x in vals if x >= 1])
    return f"{v:0.2f} {postfixes[vals.index(v)]}"

def folder_size_as_str(path: str) -> str:
    """Return folder size formatted as a string"""
    return size_to_str(folder_size(path))


class Timer():
    def __init__(self):
        self.data = {}
        self.start()
    
    def start(self):
        self.start_time = time.time()

    def split(self,split_name):
        try:
            self.data[split_name] += time.time() - self.start_time
        except KeyError:
            self.data[split_name] = time.time() - self.start_time
        self.start()

    def __repr__(self):
        out = "Times:"
        tot = 0
        for item in self.data:
            tot += self.data[item]
        for item in self.data:
            out += f"\n\t{item}: {self.data[item]:.3f}s {(self.data[item] * 100/tot):.2f}%"
        return out




class FancyTimer():
    def __init__(self):
        self.clear()

    def clear(self):
        self.data = {}
        self.triggers = []
        self.tot = 0
    
    def new_it(self):
        if len(self.triggers) > 1:
            self.tot += max(self.triggers) - min(self.triggers)
            self.triggers = []
    
    def start(self, event_name):
        if event_name not in self.data.keys():
            self.data[event_name] = {'total':0.0,'start_time':None}
        self.triggers.append(time.time())
        self.data[event_name]['start_time'] = time.time()

    def stop(self, event_name):
        t = time.time()
        self.triggers.append(time.time())
        self.data[event_name]['total'] += t - self.data[event_name]['start_time']
        self.data[event_name]['start_time'] = None

    def __repr__(self):
        self.new_it()
        out = f" Total Time: {self.tot:.3f}\nBreakdown:"
        for item, value in self.data.items():
            out += f"\n\t{item}:\t{value['total']:.3f}s {(value['total'] * 100/self.tot):.2f}%"
        return out


