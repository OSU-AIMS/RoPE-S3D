# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import matplotlib.pyplot as plt
import numpy as np
from robotpose.utils import str_to_arr

from ..simulation.kinematics import ForwardKinematics


def general_plot(joints, unit, given_err = None, actual = None, predicted = None, y_lim = None):

    if type(y_lim) in [int,float]:
        y_lim = (-y_lim,y_lim)

    plots = 2 if given_err is None else 1
    fig, axs = plt.subplots(len(joints),plots)

    if given_err is None:
        assert actual.shape[0] == predicted.shape[0]
        
        # Plot Raw Values
        for joint, idx in zip(joints,range(len(joints))):
            axs[idx,0].set_title(f'Raw {joint}')
            axs[idx,0].set_ylabel(f'({unit})')
            axs[idx,0].plot(actual[:,idx])
            axs[idx,0].plot(predicted[:,idx],color='purple')

        err = predicted - actual
    else:
        err = given_err


    zeros_err = np.zeros(err.shape[0])

    # Plot errors
    for joint, idx in zip(joints,range(len(joints))):

        idx = idx if given_err is not None else (idx, 1)

        axs[idx].set_title(f'{joint} Error')
        axs[idx].set_ylabel(f'({unit})')
        axs[idx].plot(zeros_err)
        axs[idx].plot(err[:,idx],color='purple')
        if y_lim is not None:
            axs[idx].set_ylim(y_lim)


    err = np.abs(err)

    avg_err = np.mean(err,0)

    err_std = np.std(err,0)
    err_med = np.median(err,0)
    err_90 = np.percentile(err, 90, 0)
    err_95 = np.percentile(err, 95, 0)
    err_99 = np.percentile(err, 99, 0)
    err_max = np.max(err,0)
    
    w = 6

    print(f"\nErr Stats ({unit}):")
    print(f"\t   {' '*(w-4)}Mean {' '*(w-3)}Std | {' '*(w-3)}Med {' '*(w-4)}90th {' '*(w-4)}95th {' '*(w-4)}99th {' '*(w-3)}Max")
    for joint, idx in zip(joints,range(len(joints))):
        print(f"\t{joint}: {avg_err[idx]:{w}.2f} {err_std[idx]:{w}.2f} | {err_med[idx]:{w}.2f} {err_90[idx]:{w}.2f} {err_95[idx]:{w}.2f} {err_99[idx]:{w}.2f} {err_max[idx]:{w}.2f}")

    plt.show()




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

        general_plot(self.joints,'deg',actual=self.true,predicted=self.predictions,y_lim=y_lim)

        # fig, axs = plt.subplots(len(self.joints),2)
                
        # # Plot Raw Angles
        # for joint, idx in zip(self.joints,range(len(self.joints))):
        #     axs[idx,0].set_title(f'Raw {joint} Angle')
        #     axs[idx,0].plot(self.true[:,idx])
        #     axs[idx,0].plot(self.predictions[:,idx],color='purple')

        # err = self.predictions - self.true
        # zeros_err = np.zeros(err.shape[0])

        # # Plot errors
        # for joint, idx in zip(self.joints,range(len(self.joints))):
        #     axs[idx,1].set_title(f'Angle {joint} Error')
        #     axs[idx,1].plot(zeros_err)
        #     axs[idx,1].plot(err[:,idx],color='purple')
        #     if y_lim is not None:
        #         axs[idx,1].set_ylim([-y_lim,y_lim])

        # err = np.abs(err)

        # avg_err = np.mean(err,0)

        # err_std = np.std(err,0)
        # err_med = np.median(err,0)
        # err_90 = np.percentile(err, 90, 0)
        # err_95 = np.percentile(err, 95, 0)
        # err_99 = np.percentile(err, 99, 0)
        
        # w = 5

        # print("\nStats (deg):")
        # print(f"\t   {' '*(w-4)}Mean {' '*(w-3)}Std {' '*(w-3)}Med {' '*(w-4)}90th {' '*(w-4)}95th {' '*(w-4)}99th")
        # for joint, idx in zip(self.joints,range(len(self.joints))):
        #     print(f"\t{joint}: {avg_err[idx]:{w}.2f} {err_std[idx]:{w}.2f} {err_med[idx]:{w}.2f} {err_90[idx]:{w}.2f} {err_95[idx]:{w}.2f} {err_99[idx]:{w}.2f}")

        # plt.show()






class JointDistance(ForwardKinematics):
    def __init__(self):
        super().__init__()
        self.joints_str = 'LURBT'
        self.joints = [x for x in self.joints_str]

    def distance(self, predicted: np.ndarray, actual: np.ndarray):
        assert predicted.shape[0] == actual.shape[0]

        distances = np.zeros(predicted.shape)

        for idx in range(predicted.shape[0]):
            actual_coords = self.calc(actual[idx])[1:,:3,3]
            predicted_coords = self.calc(predicted[idx])[1:,:3,3]

            distances[idx] = (np.sum((actual_coords - predicted_coords) ** 2,-1)) ** 0.5

        return distances

    def plot(self,predicted: np.ndarray, actual: np.ndarray, y_lim = None):
        err = self.distance(predicted,actual)
        general_plot(self.joints,'cm',given_err=err[:,str_to_arr(self.joints_str)]*100, y_lim=[0,y_lim*100])
