from robotpose.urdf import URDFReader
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
        err_history = np.zeros(self.history_length)

        for i in range(40):
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

            err_history[1:] = err_history[:-1]
            err_history[0] = min(over_err, under_err)
            if abs(np.mean(err_history) - err_history[0])/err_history[0] < .01:
                break

        return angles



    def _downsample(self, base, factor):
        dims = [x//factor for x in base.shape[0:2]]
        dims.reverse()
        return cv2.resize(base, tuple(dims), interpolation=cv2.INTER_LANCZOS4)

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




class AreaMatcherStaged():
    def __init__(self, camera_pose = CAMERA_POSE, ds_factor = 8):
        self.camera_pose = camera_pose
        self.ds_factor = ds_factor
        self.do_angle = np.array([True,True,True,False,False,False])
        self.angle_learning_rate = np.array([1,1,.75,.5,.5,2])
        self.history_length = 8

        self.u_reader = URDFReader()
        self.renderer = SkeletonRenderer('BASE','seg',camera_pose,f'1280_720_color_{self.ds_factor}')


    def run(self, target_img, target_depth, camera_pose = None):
        if camera_pose is None:
            camera_pose = self.camera_pose
        self.renderer.setCameraPose(camera_pose)

        target_img = self._downsample(target_img, self.ds_factor)
        target_depth = self._downsample(target_depth, self.ds_factor)

        angles = np.array([0,0.2,1.25,0,0,0], dtype=float)
        self.renderer.setJointAngles(angles)

        angle_learning_rate = np.zeros(6)

        history = np.zeros((self.history_length, 6))
        err_history = np.zeros(self.history_length)


        # Stages in form:
        # Sweep:
        #   Divisions, joints to render, angles to edit
        # Descent: 
        #   Iterations, joints to render, rate reduction, early stop thresh, angles to edit, inital learning rate
        # Flip: 
        #   joints to render, edit_angles
        l_sweep = ['sweep', 15, 3, [False,True,False,False,False,False]]
        sl_stage = ['descent',30,3,0.5,.1,[True,True,False,False,False,False],[1.2,.3,0.1,0.5,0.5,0.5]]
        u_sweep = ['sweep', 20, 6, [False,False,True,False,False,False]]
        u_stage = ['descent',20,6,0.5,.1,[True,True,True,False,False,False],[None,None,None,None,None,None]]
        s_flip_check = ['flip',6,[True,False,False,False,False,False]]
        s_check = ['descent',3,6,0.5,.05,[True,False,False,False,False,False],[.1,None,None,None,None,None]]
        lu_fine_tune = ['descent',5,6,0.5,.01,[True,True,True,False,False,False],[None,.01,.01,None,None,None]]

        stages = [l_sweep, sl_stage, u_sweep, u_stage, s_flip_check, s_check, lu_fine_tune]

        for stage in stages:

            if stage[0] == 'descent':

                for i in range(6):
                    if stage[6][i] is not None:
                        angle_learning_rate[i] = stage[6][i]

                do_ang = np.array(stage[5])

                for i in range(stage[1]):
                    for idx in np.where(do_ang)[0]:
                        if abs(np.mean(history,0)[idx] - angles[idx]) <= angle_learning_rate[idx]:
                            angle_learning_rate[idx] *= stage[3]

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

                    err_history[1:] = err_history[:-1]
                    err_history[0] = min(over_err, under_err)
                    if abs(np.mean(err_history) - err_history[0])/err_history[0] < stage[4]:
                        break

            elif stage[0] == 'flip':

                do_ang = np.array(stage[2])
                self.renderer.setMaxParts(stage[1])

                for idx in np.where(do_ang)[0]:
                    temp = angles.copy()
                    temp[idx] *= -1
                    self.renderer.setJointAngles(temp)
                    color, depth = self.renderer.render()
                    err = self._total_err(target_img, target_depth, color, depth)

                    if err < err_history[0]:
                        angles[idx] *= -1
                    
                    history[1:] = history[:-1]
                    history[0] = angles
                    err_history[1:] = err_history[:-1]
                    err_history[0] = min(err_history[1],err)

            elif stage[0] == 'sweep':
                do_ang = np.array(stage[3])
                self.renderer.setMaxParts(stage[2])
                div = stage[1]

                for idx in np.where(do_ang)[0]:
                    temp_low = angles.copy()
                    temp_low[idx] = self.u_reader.joint_limits[idx,0]
                    temp_high = angles.copy()
                    temp_high[idx] = self.u_reader.joint_limits[idx,1]

                    space = np.linspace(temp_low, temp_high, div)
                    space_err = []
                    for angs in space:
                        self.renderer.setJointAngles(angs)
                        color, depth = self.renderer.render()
                        space_err.append(self._total_err(target_img, target_depth, color, depth))

                    angles = space[space_err.index(min(space_err))]


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