# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from robotpose.urdf import URDFReader
from robotpose.simulation.render import SkeletonRenderer
from robotpose import Dataset
import numpy as np
import cv2
from robotpose.turbo_colormap import color_array
import matplotlib.pyplot as plt
from tqdm import tqdm
from robotpose.experimental.area import AreaMatcherStagedZonedError

#https://stackoverflow.com/questions/61495735/unable-to-load-numpy-formathandler-accelerator-from-opengl-accelerate

# def mask_err(target, render):
#     # Returns 0-1 (1 is bad, 0 is good)
#     target_mask = ~(np.all(target == [0, 0, 0], axis=-1))
#     render_mask = ~(np.all(render == [0, 0, 0], axis=-1))

#     # Take IOU of arrays
#     overlap = target_mask*render_mask # Logical AND
#     union = target_mask + render_mask # Logical OR
#     iou = overlap.sum()/float(union.sum())
#     return 1 - iou
    

# def depth_err(target, render):
#     target_mask = target != 0
#     render_masked = render * target_mask
#     diff = target - render_masked
#     diff = np.abs(diff) ** 0.5
#     err = np.mean(diff[diff!=0])
#     return err


# def total_err(tgt_color, tgt_depth, render_color, render_depth):
#     return 5*depth_err(tgt_depth,render_depth) + mask_err(tgt_color, render_color)


# def downsample(base, factor):
#     dims = [x//factor for x in base.shape[0:2]]
#     dims.reverse()
#     return cv2.resize(base, tuple(dims), interpolation=cv2.INTER_LANCZOS4)


# def show(color, depth, target_depth):
#     size = color.shape[0:2]
#     dim = [x*2 for x in size]
#     dim.reverse()
#     dim = tuple(dim)
#     color = cv2.resize(color, dim, interpolation=cv2.INTER_NEAREST)
#     depth = cv2.resize(depth, dim, interpolation=cv2.INTER_NEAREST)
#     target_depth = cv2.resize(target_depth, dim)
#     cv2.imshow("Color",color)
#     d = cv2.addWeighted(color_array(target_depth), .5, color_array(depth),.5,0)

#     cv2.imshow("Depth",d)
#     cv2.waitKey(1)



# CAMERA_POSE = [.042,-1.425,.399, -.01,1.553,-.057]
# WIDTH = 800


# renderer = SkeletonRenderer('BASE','seg',CAMERA_POSE,'1280_720_color_8')
# ds = Dataset('set10')

# idx = 614
# true = ds.angles[idx]

# print(true)

# ds_factor = 8
# roi_start = np.copy(ds.rois[idx,1])
# target_img = np.zeros((720,1280,3),np.uint8)
# target_img[:,roi_start:roi_start+WIDTH] = np.copy(ds.seg_img[idx])
# target_depth = np.zeros((720,1280))
# target_depth[:,roi_start:roi_start+WIDTH] = np.copy(ds.pointmaps[idx,...,2])

# target = cv2.addWeighted(target_img, .5, color_array(target_depth),.5,0)

# cv2.imshow("target",target)
# cv2.waitKey(1)

# if True:
#     target_img = downsample(target_img, ds_factor)
#     target_depth = downsample(target_depth, ds_factor)

# renderer.setJointAngles([0,0,0,0,0,0])

# dp_err = []
# mk_err = []
# s_ang = []

# hist_length = 7

# history = np.zeros((hist_length, 6))
# err_history = np.zeros(hist_length)
# err_history[err_history == 0] = np.inf

# do_angle = np.array([True,True,True,False,False,False])
# angle_learning_rate = np.array([1.2,1.2,.75,.5,.5,2])

# angles = np.array([0,0.2,1.2,0,0,0])

# renderer.setJointAngles(angles)

# u_reader = URDFReader()


# # Stages in form:
# # Descent: 
# #   Iterations, joints to render, rate reduction, early stop thresh, edit_angles, inital learning rate
# # Flip: 
# #   joints to render, edit_angles
# s_sweep = ['sweep', 10, 2, [True,False,False,False,False,False]]
# l_sweep = ['sweep', 10, 3, [False,True,False,False,False,False]]
# sl_stage = ['descent',30,3,0.5,.1,[True,True,False,False,False,False],[1.2,.3,0.1,0.5,0.5,0.5]]
# u_sweep = ['sweep', 15, 6, [False,False,True,False,False,False]]
# u_stage = ['descent',20,6,0.5,.01,[True,True,True,False,False,False],[None,None,None,None,None,None]]
# s_flip_check = ['flip',6,[True,False,False,False,False,False]]
# s_check = ['descent',5,6,0.5,.01,[True,False,False,False,False,False],[.1,None,None,None,None,None]]
# lu_fine_tune = ['descent',5,6,0.5,.01,[True,True,True,False,False,False],[None,.01,.01,None,None,None]]

# stages = [s_sweep, l_sweep, sl_stage, u_sweep, u_stage, s_flip_check, s_check, lu_fine_tune]

# for stage in stages:

#     if stage[0] == 'descent':

#         for i in range(6):
#             if stage[6][i] is not None:
#                 angle_learning_rate[i] = stage[6][i]

#         do_ang = np.array(stage[5])
#         renderer.setMaxParts(stage[2])

#         for i in range(stage[1]):
#             for idx in np.where(do_ang)[0]:
#                 if abs(np.mean(history,0)[idx] - angles[idx]) <= angle_learning_rate[idx]:
#                     angle_learning_rate[idx] *= stage[3]

#                 temp = angles.copy()
#                 temp[idx] -= angle_learning_rate[idx]
#                 renderer.setJointAngles(temp)
#                 color, depth = renderer.render()
#                 under_err = total_err(target_img, target_depth, color, depth)

#                 temp = angles.copy()
#                 temp[idx] += angle_learning_rate[idx]
#                 renderer.setJointAngles(temp)
#                 color, depth = renderer.render()
#                 over_err = total_err(target_img, target_depth, color, depth)

#                 if over_err < under_err:
#                     angles[idx] += angle_learning_rate[idx]
#                 else:
#                     angles[idx] -= angle_learning_rate[idx]

#                 # Evaluate
#                 renderer.setJointAngles(angles)
#                 color, depth = renderer.render()
#                 show(color, depth, target_depth)
#                 dp_err.append(depth_err(target_depth,depth))
#                 mk_err.append(mask_err(target_img,color))

#             history[1:] = history[:-1]
#             history[0] = angles
#             err_history[1:] = err_history[:-1]
#             err_history[0] = min(over_err,under_err)
#             if abs(np.mean(err_history) - err_history[0])/err_history[0] < stage[4]:
#                 break

#     elif stage[0] == 'flip':

#         do_ang = np.array(stage[2])
#         renderer.setMaxParts(stage[1])

#         for idx in np.where(do_ang)[0]:
#             temp = angles.copy()
#             temp[idx] *= -1
#             renderer.setJointAngles(temp)
#             color, depth = renderer.render()
#             err = total_err(target_img, target_depth, color, depth)

#             if err < err_history[0]:
#                 angles[idx] *= -1
            
#             history[1:] = history[:-1]
#             history[0] = angles
#             err_history[1:] = err_history[:-1]
#             err_history[0] = min(err_history[1],err)

#     elif stage[0] == 'sweep':
#                 do_ang = np.array(stage[3])
#                 renderer.setMaxParts(stage[2])
#                 div = stage[1]

#                 for idx in np.where(do_ang)[0]:
#                     temp_low = angles.copy()
#                     temp_low[idx] = u_reader.joint_limits[idx,0]
#                     temp_high = angles.copy()
#                     temp_high[idx] = u_reader.joint_limits[idx,1]

#                     space = np.linspace(temp_low, temp_high, div)
#                     space_err = []
#                     for angs in space:
#                         renderer.setJointAngles(angs)
#                         color, depth = renderer.render()
#                         space_err.append(total_err(target_img, target_depth, color, depth))

#                         # Evaluate
#                         show(color, depth, target_depth)

#                     angles = space[space_err.index(min(space_err))]




# print(np.array(angles))
# print((true - angles)*(180/np.pi))


# dp_err = 5 * np.array(dp_err)
# mk_err = np.array(mk_err)

# plt.plot(dp_err,label='Depth')
# plt.plot(mk_err,label='Shadow')
# plt.plot(dp_err + mk_err,label='Total')
# plt.legend()
# plt.show()



WIDTH = 800

am = AreaMatcherStagedZonedError(ds_factor=8, preview=True)
ds = Dataset('set10')

start = 900
end = 910

print("Copying Data...")
roi_start = np.copy(ds.rois[start:end,1])
target_imgs = np.zeros((end-start,720,1280,3),np.uint8)
target_depths = np.zeros((end-start,720,1280), np.float32)

og_imgs = np.copy(ds.og_img[start:end])
seg_img = np.copy(ds.seg_img[start:end])
dms = np.copy(ds.pointmaps[start:end,...,2])
cam_poses = np.copy(ds.camera_pose[start:end])

for i,s in zip(range(end-start),roi_start):
    target_imgs[i,:,s:s+WIDTH] = seg_img[i]
    target_depths[i,:,s:s+WIDTH] = dms[i]


out = []

for idx in range(end-start):
    cv2.imshow('Target',target_imgs[idx])
    cv2.waitKey(1)
    am.run(og_imgs[idx], target_imgs[idx], target_depths[idx], cam_poses[idx])



