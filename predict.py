# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import numpy as np
from tqdm import tqdm

from robotpose import Dataset, Predictor, Grapher

angs = 'SL'
dataset = 'set30'


ds = Dataset(dataset)
am = Predictor(ds_factor=2, camera_pose=ds.camera_pose[0], preview=False, base_intrin = ds.intrinsics, do_angles=angs, model_ds=dataset)

from functools import reduce

def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

div_size = factors(ds.length)
diffs = [abs(x - 200) for x in div_size]
div_size = [x for x in div_size if abs(x - 200) == min(diffs)][0]

out = []

with tqdm(total=ds.length,desc=f"Div Size {div_size}") as pbar:
    for start in range(0,ds.length,div_size):
        end = start+div_size

        #target_depths = np.zeros((div_size,720,1280))

        og_imgs = np.copy(ds.og_img[start:end])
        dms = np.copy(ds.depthmaps[start:end])
        cam_poses = np.copy(ds.camera_pose[start:end])

        for idx in range(div_size):
            out.append(am.run(og_imgs[idx], dms[idx], cam_poses[idx]))
            pbar.update(1)

out = np.array(out)

np.save(f'predictions_{dataset}.npy',out)

g = Grapher(angs,out,np.copy(ds.angles))
g.plot()
g.plot(20)
g.plot(10)
