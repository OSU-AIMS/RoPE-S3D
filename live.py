from robotpose import Predictor, JSONCoupling, LiveCamera, Dataset, Intrinsics
import numpy as np
import time
from robotpose.utils import color_array
import cv2

RUN_FOR = 60

parent_ds = 'set50'
base_intrin_str = '1280_720_color'
ds_factor = 4


base_intrin = Intrinsics(base_intrin_str)
ds = Dataset(parent_ds)

start_time = time.time()

cam = LiveCamera(base_intrin.width, base_intrin.height)
# link = JSONCoupling()
# pred = Predictor(ds.camera_pose[0],2,False,None,'SLU',base_intrin=base_intrin_str,model_ds=parent_ds)


cam.start()

while time.time() - start_time < RUN_FOR:
    # claimed = link.get_pose()
    color, depth = cam.get()
    cv2.imshow("Depth",color_array(depth))
    cv2.imshow("Color",color)
    cv2.waitKey(1)
    # calculated = pred.run(color, depth)
    # link.reset()

    # print(f"Claimed:{claimed}")
    # print(f"Calc'ed:{calculated}")
    # print(f"Diff(deg): {180*np.abs(np.array(calculated) - np.array(claimed))/np.pi}")

cam.stop()