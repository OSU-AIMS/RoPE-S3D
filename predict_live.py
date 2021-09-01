from robotpose import Predictor, JSONCoupling, LiveCamera, Dataset, Intrinsics
import numpy as np
import time
from robotpose.utils import color_array
import cv2

RUN_FOR = 6000

parent_ds = 'set63'
base_intrin_str = '1280_720_color'
ds_factor = 8


base_intrin = Intrinsics(base_intrin_str)
ds = Dataset(parent_ds)

cam = LiveCamera(base_intrin.width, base_intrin.height)
link = JSONCoupling()
pred = Predictor(ds.camera_pose[0],8,False,None,'SLU',base_intrin=base_intrin_str,model_ds=parent_ds)


claims = []
predictions = []

cam.start()
start_time = time.time()
while time.time() - start_time < RUN_FOR:
    print("Ready")
    claimed = link.get_pose()
    color, depth = cam.get()
    cv2.imshow("Depth",color_array(depth))
    cv2.imshow("Color",color)
    cv2.waitKey(1)
    calculated = pred.run(color, depth)
    link.reset()

    print(type(claimed))

    print("")
    print(f"Claimed (r):{[f'{x:1.4f}' for x in claimed]}")
    print(f"Calc'ed (r):{[f'{x:1.4f}' for x in calculated]}")
    print(f"Diff(deg): {[f'{x:1.2f}' for x in 180*np.abs(np.array(calculated) - np.array(claimed))/np.pi]}")

    claims.append(claimed)
    predictions.append(calculated)

    out = np.stack((np.array(claims),np.array(predictions)))
    np.save('live_preds.npy',out)

    

cam.stop()