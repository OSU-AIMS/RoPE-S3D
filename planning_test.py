from robotpose.training.planning import Planner
from robotpose import DatasetRenderer
import cv2
import numpy as np

r = DatasetRenderer('set40')
p = Planner()

a = [x *.1 for x in range(10)]

divs = [a, a, a,[0],[-1,0,1],[0]]

# for pose in p.noisyGrid('SLU',500,[.1,.2,.1,0,0,0]):
#     r.setJointAngles(pose)
#     color, depth = r.render()
#     cv2.imshow("",color)
#     cv2.waitKey(1)

np.save('plan.npy',p.basicGrid('SLU',300))