from robotpose.prediction.analysis import JointDistance
from robotpose import Predictor, JSONCoupling, LiveCamera, Dataset, Intrinsics
import numpy as np
from robotpose.utils import color_array
import cv2
from tqdm import tqdm
import logging


LENGTH = 3
ALLOWED_DEVIANCE = 0.2
MAX_PBAR = int(ALLOWED_DEVIANCE * 1000)


# class Live():
#     def __init__(self, base_intrin_str, parent_ds, angs, ds_factor) -> None:
#         base_intrin = Intrinsics(base_intrin_str)
#         ds = Dataset(parent_ds)

#         self.cam = LiveCamera(base_intrin.width, base_intrin.height)
#         self.link = JSONCoupling()
#         self.pred = Predictor(ds.camera_pose[0],ds_factor,False,None,angs,base_intrin=base_intrin_str,model_ds=parent_ds)

#         self.cam.start()

#         self.claims = np.zeros((LENGTH,6))
#         self.predictions = np.zeros((LENGTH,6))

#         self.create_pbars()

#     def stop(self):
#         self.cam.stop()

#     def run(self):
#         logging.info("Ready")
#         # off = -0.05
#         while True:
#             claimed = self.link.get_pose()

#             # claimed[0] += off
#             # off += 0.05

#             color, depth = self.cam.get()
#             cv2.imshow("Depth",color_array(depth))
#             cv2.imshow("Color",color)
#             cv2.waitKey(1)
#             calculated = self.pred.run(color, depth)
#             self.link.reset()

#             self.shift_in(claimed, calculated)
#             self.update_error()
#             self.update_pbars()
#             self.displayState()


#     def shift_in(self, claim, prediction):
#         self.claims[1:] = self.claims[:-1]
#         self.predictions[1:] = self.predictions[:-1]
#         self.claims[0] = claim
#         self.predictions[0] = prediction

#     def create_pbars(self):
#         titles = [x for x in 'SLURBT']
#         self.pbars = [tqdm(desc=titles[pos],unit='mRad',position=pos,leave=False, total=MAX_PBAR) for pos in range(len(titles))]

#     def update_error(self):
#         self.diff = np.abs(self.claims - self.predictions)
#         self.out_of_range = self.diff > ALLOWED_DEVIANCE
#         # print((self.diff[0] * 180 / np.pi).astype(int))

#     def update_pbars(self):
#         for idx in range(len(self.pbars)):
#             self.pbars[idx].colour = 'red' if self.out_of_range[0,idx] else 'green'
#             self.pbars[idx].n = np.clip(int(self.diff[0,idx] * 1000),0,MAX_PBAR-1)
#             self.pbars[idx].refresh()

#     @property
#     def state(self):
#         return np.sum(np.prod(self.out_of_range,0)) > 0

#     def displayState(self):
#         a = np.zeros((500,500,3),np.uint8)
#         if self.state:
#             a[...,2] = 255
#         else:
#             a[...,1] = 255
#         cv2.imshow("State",a)
#         cv2.waitKey(1)


LENGTH = 3
ALLOWED_DEVIANCE = 0.1
MAX_PBAR = int(ALLOWED_DEVIANCE * 1000)


class Live():
    def __init__(self, base_intrin_str, parent_ds, angs, ds_factor) -> None:
        base_intrin = Intrinsics(base_intrin_str)
        ds = Dataset(parent_ds)

        self.cam = LiveCamera(base_intrin.width, base_intrin.height)
        self.link = JSONCoupling()
        self.pred = Predictor(ds.camera_pose[0],ds_factor,False,None,angs,base_intrin=base_intrin_str,model_ds=parent_ds)
        self.jd = JointDistance()

        self.cam.start()

        self.claims = np.zeros((LENGTH,6))
        self.predictions = np.zeros((LENGTH,6))

        self.running_claims = []
        self.running_predictions = []

        self.create_pbar()

    def stop(self):
        self.cam.stop()

    def run(self):
        logging.info("Ready")
        while True:
            claimed = self.link.get_pose()

            color, depth = self.cam.get()
            cv2.imshow("Depth",color_array(depth))
            cv2.imshow("Color",color)
            cv2.waitKey(1)
            calculated = self.pred.run(color, depth)
            self.link.reset()

            self.shift_in(claimed, calculated)
            self.update_error()
            self.update_pbar()
            self.displayState()
            self.save()


    def shift_in(self, claim, prediction):
        self.claims[1:] = self.claims[:-1]
        self.predictions[1:] = self.predictions[:-1]
        self.claims[0] = claim
        self.predictions[0] = prediction
        self.running_claims.append(claim)
        self.running_predictions.append(prediction)

    def create_pbar(self):
        self.pbar = tqdm(desc='TCP',unit='mm',leave=False, total=MAX_PBAR)

    def update_error(self):
        self.diff = self.jd.single(self.predictions,self.claims)
        self.out_of_range = self.diff > ALLOWED_DEVIANCE
        
    def update_pbar(self):
        self.pbar.colour = 'red' if self.out_of_range[0] else 'green'
        self.pbar.n = np.clip(int(self.diff[0] * 1000),0,MAX_PBAR-1)
        self.pbar.refresh()

    def save(self):
        c = np.array(self.running_claims)
        p = np.array(self.running_predictions)
        a = np.zeros((2,*c.shape))
        a[0] = c
        a[1] = p
        np.save('live_preds.npy',a)

    @property
    def state(self):
        return np.sum(self.out_of_range,0) == LENGTH

    def displayState(self):
        a = np.zeros((500,500,3),np.uint8)
        if self.state:
            a[...,2] = 255
        else:
            a[...,1] = 255
        cv2.imshow("State",a)
        cv2.waitKey(1)



if __name__ == "__main__":
    a = Live('1280_720_color','set91','SLU',8)
    a.run()