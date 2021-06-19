# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import multiprocessing as mp
import string
import time
import subprocess as sp

import cv2
import matplotlib.pyplot as plt
import numpy as np


def setMemoryGrowth():
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def get_gpu_memory():
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
    memory_free_values = [int(x.split()[0])*67108864 for i, x in enumerate(memory_free_info)]
    return memory_free_values 


def workerCount():
    cpu_count = mp.cpu_count()
    return int(min(cpu_count - 2, .75 * cpu_count))


def expandRegion(image, size, iterations = 1):
    kern = np.ones((size,size), dtype=np.uint8)
    return cv2.dilate(image, kern, iterations = iterations)


def str_to_arr(string):
    joints = ['S','L','U','R','B','T']
    out = np.zeros(6, bool)
    for letter in string.upper():
        out[joints.index(letter)] = True
    return out

def get_key(dict, val):
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



class Grapher():

    def __init__(self, joints_to_plot: str, predictions: np.ndarray, ds_angles: np.ndarray = None):
        self.compare = ds_angles is not None
        self.joints = [x for x in joints_to_plot.upper()]
        self.predictions = np.degrees(predictions)
        self.true = np.degrees(ds_angles)
        self._b_correction()
        self._cropComparison()
    
    def plot(self,ylim=None):
        self._plotWithComparison(ylim)

    def _b_correction(self):
        if 'B' not in self.joints:
            return

        offsets = [-360, -180, 0, 180, 360]
        
        for idx in range(len(self.predictions)):
            err = [abs((self.predictions[idx,4] + x) - self.true[idx,4]) for x in offsets]
            self.predictions[idx,4] += offsets[err.index(min(err))]


    def _cropComparison(self):
        ang = ['S','L','U','R','B','T']
        l = len(self.predictions)
        true = np.copy(self.true)
        predictions = np.copy(self.predictions)
        self.true = np.zeros((l,len(self.joints)))
        self.predictions = np.zeros((l,len(self.joints)))
        for joint, idx in zip(self.joints,range(len(self.joints))):
            self.true[:,idx] = true[:l,ang.index(joint)]
            self.predictions[:,idx] = predictions[:l,ang.index(joint)]

    def _plotWithComparison(self, y_lim = None):

        fig, axs = plt.subplots(len(self.joints),2)
                
        # Plot Raw Angles
        for joint, idx in zip(self.joints,range(len(self.joints))):
            axs[idx,0].set_title(f'Raw {joint} Angle')
            axs[idx,0].plot(self.true[:,idx])
            axs[idx,0].plot(self.predictions[:,idx],color='purple')

        err = self.predictions - self.true
        zeros_err = np.zeros(err.shape[0])

        # Plot errors
        for joint, idx in zip(self.joints,range(len(self.joints))):
            axs[idx,1].set_title(f'Angle {joint} Error')
            axs[idx,1].plot(zeros_err)
            axs[idx,1].plot(err[:,idx],color='purple')
            if y_lim is not None:
                axs[idx,1].set_ylim([-y_lim,y_lim])

        err = np.abs(err)

        avg_err = np.mean(err,0)

        err_std = np.std(err,0)
        err_med = np.median(err,0)
        err_90 = np.percentile(err, 90, 0)
        err_95 = np.percentile(err, 95, 0)
        err_99 = np.percentile(err, 99, 0)
        w = 5

        print("\nStats (deg):")
        print(f"\t   {' '*(w-4)}Mean {' '*(w-3)}Std {' '*(w-3)}Med {' '*(w-4)}90th {' '*(w-4)}95th {' '*(w-4)}99th")
        for joint, idx in zip(self.joints,range(len(self.joints))):
            print(f"\t{joint}: {avg_err[idx]:{w}.2f} {err_std[idx]:{w}.2f} {err_med[idx]:{w}.2f} {err_90[idx]:{w}.2f} {err_95[idx]:{w}.2f} {err_99[idx]:{w}.2f}")

        plt.show()
