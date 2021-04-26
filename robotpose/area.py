from robotpose.simulation.render import SkeletonRenderer
import numpy as np
import cv2


CAMERA_POSE = [.042,-1.425,.399, -.01,1.553,-.057]
WIDTH = 800


class AreaMatcher():
    def __init__(self, camera_pose = CAMERA_POSE, ds_factor = 8):
        self.camera_pose = camera_pose
        self.ds_factor = ds_factor
        self.do_angle = np.array([True,True,True,False,False,False])
        self.angle_learning_rate = np.array([1,1,.75,.5,.5,2])
        self.history_length = 8

        self.renderer = SkeletonRenderer('BASE','seg',camera_pose,f'1280_720_color_{self.ds_factor}')


    def run(self, target_img, target_depth, camera_pose = None):
        if camera_pose is None:
            camera_pose = self.camera_pose
        self.renderer.setCameraPose(camera_pose)

        target_img = self._downsample(target_img, self.ds_factor)
        target_depth = self._downsample(target_depth, self.ds_factor)

        angles = np.array([0,0.2,1.25,0,0,0], dtype=float)
        self.renderer.setJointAngles(angles)

        angle_learning_rate = np.copy(self.angle_learning_rate)

        history = np.zeros((self.history_length, 6))    

        for i in range(30):
            for idx in np.where(self.do_angle)[0]:

                if abs(np.mean(history,0)[idx] - angles[idx]) <= angle_learning_rate[idx]:
                    angle_learning_rate[idx] *= .5

                temp = angles.copy()
                temp[idx] -= angle_learning_rate[idx]

                # Under
                self.renderer.setJointAngles(temp)
                color, depth = self.renderer.render()
                under_err = self._total_err(target_img, target_depth, color, depth)

                # Over
                temp = angles.copy()
                temp[idx] += angle_learning_rate[idx]
                self.renderer.setJointAngles(temp)
                color, depth = self.renderer.render()
                over_err = self._total_err(target_img, target_depth, color, depth)

                if over_err < under_err:
                    angles[idx] += angle_learning_rate[idx]
                else:
                    angles[idx] -= angle_learning_rate[idx]


            history[1:] = history[:-1]
            history[0] = angles

        return angles



    def _downsample(self, base, factor):
        dims = [x//factor for x in base.shape[0:2]]
        dims.reverse()
        return cv2.resize(base, tuple(dims))

    def _mask_err(self, target, render):
        # Returns 0-1 (1 is bad, 0 is good)
        target_mask = ~(np.all(target == [0, 0, 0], axis=-1))
        render_mask = ~(np.all(render == [0, 0, 0], axis=-1))

        # Take IOU of arrays
        overlap = target_mask*render_mask # Logical AND
        union = target_mask + render_mask # Logical OR
        iou = overlap.sum()/float(union.sum())
        return 1 - iou     

    
    def _depth_err(self, target, render):
        target_mask = target != 0
        render_masked = render * target_mask
        diff = target - render_masked
        diff = np.abs(diff) ** 0.5
        err = np.mean(diff[diff!=0])
        return err

    def _total_err(self, tgt_color, tgt_depth, render_color, render_depth):
        return 5*self._depth_err(tgt_depth,render_depth) + self._mask_err(tgt_color, render_color)


