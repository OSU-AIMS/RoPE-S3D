from robotpose.simulation.render import SkeletonRenderer
from robotpose import Dataset
import numpy as np
import cv2
from robotpose.turbo_colormap import color_array
import matplotlib.pyplot as plt
from tqdm import tqdm

def mask_err(target, render):
    # Returns 0-1 (1 is bad, 0 is good)
    target_mask = ~(np.all(target == [0, 0, 0], axis=-1))
    render_mask = ~(np.all(render == [0, 0, 0], axis=-1))

    # Take IOU of arrays
    overlap = target_mask*render_mask # Logical AND
    union = target_mask + render_mask # Logical OR
    iou = overlap.sum()/float(union.sum())
    return 1 - iou
    

def depth_err(target, render):
    target_mask = target != 0
    render_masked = render * target_mask
    diff = target - render_masked
    diff = np.abs(diff) ** 0.5
    err = np.mean(diff[diff!=0])
    return err

def downsample(base, factor):
    dims = [x//factor for x in base.shape[0:2]]
    dims.reverse()
    return cv2.resize(base, tuple(dims))


CAMERA_POSE = [.042,-1.425,.399, -.01,1.553,-.057]
WIDTH = 800


renderer = SkeletonRenderer('BASE','seg',CAMERA_POSE,'1280_720_color_8')
ds = Dataset('set10')

idx = 20

print(ds.angles[idx])

ds_factor = 8
roi_start = np.copy(ds.rois[idx,1])
target_img = np.zeros((720,1280,3),np.uint8)
target_img[:,roi_start:roi_start+WIDTH] = np.copy(ds.seg_img[idx])
target_depth = np.zeros((720,1280))
target_depth[:,roi_start:roi_start+WIDTH] = np.copy(ds.pointmaps[idx,...,2])

if True:
    target_img = downsample(target_img, ds_factor)
    target_depth = downsample(target_depth, ds_factor)

renderer.setJointAngles([0,0,0,0,0,0])

dp_err = []
mk_err = []
s_ang = []
l_ang = []

do_angle = np.array([True,True,False,False,False,False])
angle_learning_rate = np.array([.01,.01,.1,.1,.1,.1])

angles = [-0.4,.5,0,0,0,0]

renderer.setJointAngles(angles)

for i in range(100):
    for idx in np.where(do_angle)[0]:
        temp = angles.copy()
        temp[idx] -= angle_learning_rate[idx]

        # Under
        renderer.setJointAngles(temp)
        color, depth = renderer.render()
        under_err = depth_err(target_depth,depth) + mask_err(target_img,color)

        # Over
        temp = angles.copy()
        temp[idx] += angle_learning_rate[idx]
        renderer.setJointAngles(temp)
        color, depth = renderer.render()
        over_err = depth_err(target_depth,depth) + mask_err(target_img,color)

        if over_err < under_err:
            angles[idx] += angle_learning_rate[idx]
        else:
            angles[idx] -= angle_learning_rate[idx]

        # Evaluate
        renderer.setJointAngles(angles)
        color, depth = renderer.render()
        cv2.imshow("test",color)
        cv2.waitKey(1)
        dp_err.append(depth_err(target_depth,depth))
        mk_err.append(mask_err(target_img,color))
        s_ang.append(angles[idx])

print(np.array(angles))

dp_err = np.array(dp_err)
mk_err = np.array(mk_err)

plt.plot(dp_err,label='Depth')
plt.plot(mk_err,label='Shadow')
plt.plot(dp_err + mk_err,label='Total')
plt.legend()
plt.show()


# cv2.imshow("test", color_array(target_depth-depth))
# print(np.mean(target_depth))
# print(np.mean(depth))
# cv2.waitKey(0)




