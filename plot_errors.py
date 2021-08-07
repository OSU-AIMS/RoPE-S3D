from robotpose.simulation.render import DatasetRenderer
import robotpose
import numpy as np
from robotpose.utils import Grapher
from robotpose import Dataset
import cv2

# dataset = 'set30'


# ds = Dataset(dataset)

# preds = np.load(f'predictions_{dataset}.npy')
# angles = np.copy(ds.angles)

results = np.load('synth_test.npy')
angles = results[0]
preds = results[1]


IDX_TO_USE = 0
PERCENTILE_TO_SHOW = 99




indicies = np.argsort(angles[...,IDX_TO_USE])

out = np.sort(indicies)
# print(out)

g = Grapher('SLU',preds[indicies],angles[indicies])
g.plot(20)

diff = np.abs(preds - angles)
#print(diff)

#print((diff[:,0] > np.percentile(diff[:,IDX_TO_USE],PERCENTILE_TO_SHOW)))


# idxs = np.where(diff[:,IDX_TO_USE] > np.percentile(diff[:,IDX_TO_USE],PERCENTILE_TO_SHOW))
# imgs = np.copy(ds.og_img[idxs])
# preds, diff = preds[idxs], diff[idxs]

# r = DatasetRenderer(dataset)
# r.setMaxParts(6)

# for idx in range(len(imgs)):
#     r.setJointAngles(preds[idx])
#     c,d = r.render()
#     out = cv2.addWeighted(imgs[idx],.5,c,.5,0)
#     print(diff[idx] * 180 / np.pi)
#     cv2.imshow("",out)
#     cv2.waitKey(0)