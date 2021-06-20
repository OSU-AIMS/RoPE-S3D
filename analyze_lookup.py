# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley
from robotpose.simulation.lookup import RobotLookupManager
from robotpose.simulation.render import DatasetRenderer, Renderer
from robotpose import Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from robotpose.utils import get_gpu_memory


class LookupErrorViewer():
    def __init__(self, ds_name, ds_factor = 8):
        self.ds_factor = ds_factor
        self.ds = Dataset(ds_name)
        self.renderer = DatasetRenderer(ds_name)
        self.renderer.intrinsics.downscale(ds_factor)
        
        lm = RobotLookupManager()
        self.lookup_angles, self.lookup_depth = lm.get(self.renderer.intrinsics, self.ds.camera_pose[0], 4, 'SL', int(get_gpu_memory()[0] / (3 * 32)))


    def visualize(self, idx):
        target_depth = np.copy(self.ds.depthmaps[idx])
        true = self.ds.angles[idx]

        target_depth = self._downsample(target_depth, self.ds_factor)

        self._load_target(target_depth)
        self.renderer.setMaxParts(4)
        
        self.renderer.setCameraPose(self.ds.camera_pose[idx])
        self.renderer.setPosesFromDS(idx)
        c, d = self.renderer.render()
        d = self._downsample(d, self.ds_factor)
        real_mask = d != 0

        # #Old
        # diff = self._tgt_depth_stack ** 0.5 - self.lookup_depth ** 0.5
        # tf_err = tf.keras.losses.MSE(tf.constant(self._tgt_depth_stack), tf.constant(self.lookup_depth)).numpy()
        # diff = np.abs(diff)# ** 0.5
        # lookup_err = np.mean(diff, (1,2)) *-np.var(diff, (1,2))

        # New
        # diff = self._tgt_depth_stack_half - tf.pow(tf.constant(self.lookup_depth,tf.float32),0.5)
        # diff = tf.abs(diff) 
        # lookup_err = (tf.reduce_mean(diff, (1,2)) *- tf.math.reduce_std(diff, (1,2))).numpy()

        diff = self._tgt_depth_stack_full - tf.constant(self.lookup_depth,tf.float32)
        diff = tf.abs(diff)* tf.stack([tf.constant(real_mask,tf.float32)]*len(self.lookup_angles))
        lookup_err = (tf.reduce_mean(diff, (1,2)) * tf.math.reduce_std(diff, (1,2))).numpy()


        s,l = np.meshgrid(np.unique(self.lookup_angles[:,0]),np.unique(self.lookup_angles[:,1]), indexing='ij')
        err = np.zeros((len(s),len(l)))

        for s_idx in range(len(s)):
            for l_idx in range(len(l)):
                err[s_idx, l_idx] = lookup_err[s_idx + l_idx * len(s)]
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot_surface(s, l, err, rstride=1, cstride=1,cmap='viridis', edgecolor='none', zorder=1, alpha=.8)
        ax.plot([true[0]]*2, [true[1]]*2, [min(lookup_err), max(lookup_err)],'m', zorder=2)
        ax.plot([self.lookup_angles[lookup_err.argmin(),0]]*2, [self.lookup_angles[lookup_err.argmin(),1]]*2, [min(lookup_err), max(lookup_err)],'r', zorder=2)
        ax.set_xlabel('S')
        ax.set_ylabel('L')
        plt.show()


    def distance(self):
        s_err = []
        l_err = []
        dist = []

        depths = np.copy(self.ds.depthmaps)
        angs = np.copy(self.ds.angles)

        for idx in tqdm(range(self.ds.length)):
        # for idx in tqdm(range(100)):
            target_depth = depths[idx]
            true = angs[idx]

            self.renderer.setCameraPose(self.ds.camera_pose[idx])
            self.renderer.setPosesFromDS(idx)
            c, d = self.renderer.render()
            d = self._downsample(d, self.ds_factor)
            real_mask = d != 0

            target_depth = self._downsample(target_depth, self.ds_factor)

            self._load_target(target_depth)

            # # #Old
            # diff = self._tgt_depth_stack ** 0.5 - self.lookup_depth ** 0.5
            # #tf_err = tf.keras.losses.MSE(tf.constant(self._tgt_depth_stack), tf.constant(self.lookup_depth)).numpy()
            # diff = np.abs(diff)# ** 0.5
            # lookup_err = np.mean(diff, (1,2)) *-np.var(diff, (1,2))

            # New
            # diff = self._tgt_depth_stack_half - tf.pow(tf.constant(self.lookup_depth,tf.float32),0.5)
            # diff = tf.abs(diff) 
            # lookup_err = (tf.reduce_mean(diff, (1,2)) *- tf.math.reduce_std(diff, (1,2)))
            # s,l = self.lookup_angles[tf.argmin(lookup_err).numpy(),:2]


            # diff = self._tgt_depth_stack_full - tf.constant(self.lookup_depth,tf.float32)
            # diff = tf.pow(diff,2)
            # lookup_err = (tf.reduce_mean(diff, (1,2)) *- tf.math.reduce_std(diff, (1,2)))
            # s,l = self.lookup_angles[tf.argmin(lookup_err).numpy(),:2]


            diff = self._tgt_depth_stack_full - tf.constant(self.lookup_depth,tf.float32)
            diff = tf.abs(diff)* tf.stack([tf.constant(real_mask,tf.float32)]*len(self.lookup_angles))
            lookup_err = (tf.reduce_mean(diff, (1,2))).numpy()
            s,l = self.lookup_angles[tf.argmin(lookup_err).numpy(),:2]

            # s,l = self.lookup_angles[lookup_err.argmin(),:2]
            s_act,l_act = true[:2]

            s_err.append(abs(s_act - s)*180/np.pi)
            l_err.append(abs(l_act - l)*180/np.pi)
            dist.append(np.sqrt((s_act-s)**2 + (l_act-l)**2)*180/np.pi)

        plt.plot(s_err)
        plt.plot(l_err)
        plt.plot(dist)
        plt.show()




    def _downsample(self, base: np.ndarray, factor: int) -> np.ndarray:
        dims = [x//factor for x in base.shape[0:2]]
        dims.reverse()
        return cv2.resize(base, tuple(dims))

    def _load_target(self, tgt_depth: np.ndarray) -> None:
        self._tgt_depth = tgt_depth
        # self._tgt_depth_stack = np.stack([tgt_depth]*len(self.lookup_angles))
        # self._tgt_depth_stack_half = tf.stack([tf.pow(tf.constant(tgt_depth, tf.float32),2)]*len(self.lookup_angles))
        self._tgt_depth_stack_full = tf.stack([tf.constant(tgt_depth, tf.float32)]*len(self.lookup_angles))


a = LookupErrorViewer('set20',4)
a.visualize(236)
#a.distance()


