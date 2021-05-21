# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from robotpose.simulation.render import Renderer
from robotpose import Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py


class LookupErrorViewer():
    def __init__(self, ds_factor = 8):
        self.ds_factor = ds_factor
        self.renderer = Renderer('seg',None,f'1280_720_color_{self.ds_factor}')

        with h5py.File('test.h5','r') as f:
            self.lookup_angles = np.copy(f['angles'])
            self.lookup_depth = np.copy(f['depth'])


    def run(self, target_depth, camera_pose, true):
        self.renderer.setCameraPose(camera_pose)
        target_depth = self._downsample(target_depth, self.ds_factor)


        self._load_target(target_depth)
        self.renderer.setMaxParts(4)

        diff = self._tgt_depth_stack ** 0.5 - self.lookup_depth ** 0.5
        diff = np.abs(diff)# ** 0.5
        lookup_err = np.mean(diff, (1,2)) *-np.var(diff, (1,2))

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

    def _downsample(self, base: np.ndarray, factor: int) -> np.ndarray:
        dims = [x//factor for x in base.shape[0:2]]
        dims.reverse()
        return cv2.resize(base, tuple(dims))

    def _load_target(self, tgt_depth: np.ndarray) -> None:
        self._tgt_depth = tgt_depth
        self._tgt_depth_stack = np.stack([tgt_depth]*len(self.lookup_angles))


a = LookupErrorViewer()
ds = Dataset('set10')

idx = 704

a.run(np.copy(ds.depthmaps[idx]),ds.camera_pose[idx],ds.angles[idx])


