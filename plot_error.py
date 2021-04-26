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

renderer = SkeletonRenderer('BASE','seg',CAMERA_POSE)
renderer_quarter = SkeletonRenderer('BASE','seg',CAMERA_POSE,'1280_720_color_8')
ds = Dataset('set10')

idx = 20

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
renderer_quarter.setJointAngles([0,0,0,0,0,0])

dp_err = []
mk_err = []
s_ang = []
l_ang = []

ns = 50
nl = 50

err = np.zeros((ns, nl))

space_s = np.linspace(-np.pi,np.pi,ns)
space_l = np.linspace(-.1,.1,nl)

sv, lv = np.meshgrid(space_s,space_l)

for s in tqdm(range(ns)):
    for l in range(nl):
        renderer_quarter.setJointAngles([sv[s,l],lv[s,l],0,0,0,0])
        color, depth = renderer_quarter.render()

        dp_err.append(depth_err(target_depth,depth))
        mk_err.append(mask_err(target_img,color))
        err[s,l] = depth_err(target_depth,depth) + mask_err(target_img,color)

dp_err = np.array(dp_err)
mk_err = np.array(mk_err)


# plt.plot(space,dp_err,label='Depth')
# plt.plot(space,mk_err,label='Shadow')
# plt.plot(space,dp_err + mk_err,label='Total')
# plt.axvline(x=ds.angles[idx,0], color='red')
# plt.legend()
# plt.show()


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(sv, lv, err, rstride=1, cstride=1,cmap='viridis', edgecolor='none')
ax.set_xlabel('S')
ax.set_ylabel('L')
plt.show()

# cv2.imshow("test", color_array(target_depth-depth))
# print(np.mean(target_depth))
# print(np.mean(depth))
# cv2.waitKey(0)




