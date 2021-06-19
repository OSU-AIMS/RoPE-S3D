# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from robotpose import Dataset
from robotpose.prediction.camera import ModellessCameraPredictor, CameraPredictor
import numpy as np


ds = Dataset('set20')

pred = ModellessCameraPredictor([1.85,-1.1,.6,0,0,1],base_intrinsics=ds.attrs['color_intrinsics'], ds_factor=8, preview=True)#, save_to='output/projection_viz.avi')
pred_model = CameraPredictor([1.85,-1.1,.6,0,0,1], base_intrinsics=ds.attrs['color_intrinsics'], ds_factor=8, preview=True)#, save_to='output/projection_viz.avi')


# idxs = [50,167,583,901,224]
# idxs = [50,0,25,75,99]
# idxs = [x for x in range(0,100,5)]

# 20 random chosen by numpy
idxs = [205, 526, 263, 109, 774, 722, 151, 107, 485, 344, 679, 621, 694, 919, 110, 618, 367, 587, 352, 277]
#idxs = [50]

print("Copying Data...")
idxs_sorted = idxs.copy()
idxs_sorted.sort()
idx_map = [idxs.index(x) for x in idxs_sorted]

target_depth = np.copy(ds.depthmaps[idxs_sorted])
og_img = np.copy(ds.og_img[idxs_sorted])
angles = np.copy(ds.angles[idxs_sorted])

target_depth = target_depth[idx_map]
og_img = og_img[idx_map]
angles = angles[idx_map]

predicted = pred.run(np.copy(og_img), np.copy(target_depth), np.copy(angles))
# predicted = pred.run(np.copy(og_img), np.copy(target_depth), np.copy(angles), predicted)
predicted_model = pred_model.run(np.copy(og_img), np.copy(target_depth), np.copy(angles))
# predicted_model = pred_model.run(np.copy(og_img), np.copy(target_depth), np.copy(angles),predicted_model)


print(f"Actual: {ds.camera_pose[idxs[0]]}\t{pred.error_at(ds.camera_pose[idxs[0]])}\t{pred_model.error_at(ds.camera_pose[idxs[0]])}")
print(f"Predicted: {predicted}\t{pred.error_at(predicted)}\t{pred_model.error_at(predicted)}")
print(f"Predictedm: {predicted_model}\t{pred.error_at(predicted_model)}\t{pred_model.error_at(predicted_model)}")