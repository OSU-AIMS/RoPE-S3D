from robotpose.utils import Grapher, workerCount
from robotpose.area import AreaMatcher
from robotpose import Dataset
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from robotpose.simulation import SkeletonRenderer

WIDTH = 800

am = AreaMatcher(ds_factor=8)
ds = Dataset('set10')

start = 0
end = 1000

roi_start = np.copy(ds.rois[start:end,1])
target_imgs = np.zeros((end-start,720,1280,3),np.uint8)
target_depths = np.zeros((end-start,720,1280))


seg_img = np.copy(ds.seg_img[start:end])
dms = np.copy(ds.pointmaps[start:end,...,2])
cam_poses = np.copy(ds.camera_pose[start:end])

for i,s in zip(range(end-start),roi_start):
    target_imgs[i,:,s:s+WIDTH] = seg_img[i]
    target_depths[i,:,s:s+WIDTH] = dms[i]


out = []

for idx in tqdm(range(end-start)):
    out.append(am.run(target_imgs[idx], target_depths[idx], cam_poses[idx]))

out = np.array(out)

out_dicts = []
for pred in out:
    out_dict = {}
    for i,key in zip(range(3),['S','L','U']):
        out_dict[key] = {}
        out_dict[key]['val'] = pred[i]
        out_dict[key]['percent_est'] = 0
    out_dicts.append(out_dict)

g = Grapher(['S','L','U'],out_dicts,np.copy(ds.angles))
g.plot()
g.plot(20)



# inp = []

# for idx in tqdm(range(end-start)):
#     inp.append([target_imgs[idx], target_depths[idx], cam_poses[idx]])

# class Boi():
#     def __init__(self, inputs):
#         self.am = AreaMatcher(ds_factor=8)

#         with mp.Pool(workerCount()) as pool:
#             out = pool.starmap(self.am.run, inputs)

#         print(out)

# Boi(inp)