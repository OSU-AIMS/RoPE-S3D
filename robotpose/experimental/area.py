# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import numpy as np

import cv2
from numpy.lib.function_base import angle
from pixellib.instance import custom_segmentation
from scipy.interpolate import interp1d

from robotpose.simulation.render import SkeletonRenderer
from robotpose.urdf import URDFReader
from ..turbo_colormap import color_array


CAMERA_POSE = [.042,-1.425,.399, -.01,1.553,-.057]
WIDTH = 800

class AreaMatcherStagedZonedError():
    def __init__(self, camera_pose = CAMERA_POSE, ds_factor = 8, preview = False):
        self.camera_pose = camera_pose
        self.ds_factor = ds_factor
        self.do_angle = np.array([True,True,True,False,False,False])
        self.min_learning_rate = np.array([.005]*6)
        self.history_length = 5
        self.preview = preview

        self.u_reader = URDFReader()
        self.renderer = SkeletonRenderer('BASE','seg',camera_pose,f'1280_720_color_{self.ds_factor}')

        self.classes = ["BG","base_link","link_s", "link_l", "link_u","link_r","link_b"]
        self.link_names = self.classes[1:]

        self.seg = custom_segmentation()
        self.seg.inferConfig(num_classes=6, class_names=self.classes)
        self.seg.load_model("models/segmentation/multi/A.h5")


    def run(self, og_image, target_img, target_depth, camera_pose = None, starting_point = None):
        if camera_pose is None:
            camera_pose = self.camera_pose
        self.renderer.setCameraPose(camera_pose)

        target_img = self._downsample(target_img, self.ds_factor)
        target_depth = self._downsample(target_depth, self.ds_factor)
        r, output = self.seg.segmentImage(self._downsample(og_image, self.ds_factor), process_frame=True)
        segmentation_data = self._reorganize_by_link(r)

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

        # s_sweep = ['sweep', 15, 2, [True,False,False,False,False,False]]
        # l_sweep = ['sweep', 15, 3, [False,True,False,False,False,False]]
        # sl_rough = ['descent',15,3,0.5,.25,[True,True,False,False,False,False],[0.3,0.2,0.075,0.5,0.5,0.5]]
        # u_sweep = ['sweep', 15, 6, [False,False,True,False,False,False]]
        # u_stage = ['descent',20,6,0.5,.1,[True,True,True,False,False,False],[None,None,None,None,None,None]]
        # s_flip_check = ['flip',6,[True,False,False,False,False,False]]
        # s_check = ['descent',3,6,0.4,.05,[True,False,False,False,False,False],[.1,None,None,None,None,None]]
        # lu_fine_tune = ['descent',5,6,0.25,.01,[True,True,True,False,False,False],[None,None,None,None,None,None]]

        # stages = [s_sweep, l_sweep, sl_rough, u_sweep, u_stage, s_flip_check, s_check, lu_fine_tune]

        # s_sweep = ['sweep', 20, 2, [True,False,False,False,False,False]]
        # l_sweep = ['sweep', 20, 3, [False,True,False,False,False,False]]
        # s_flip_check_3 = ['flip',3,[True,False,False,False,False,False]]
        # sl_rough = ['descent',15,3,0.7,.2,[True,True,False,False,False,False],[0.8,0.5,0.075,0.5,0.5,0.5]]
        # u_sweep = ['sweep', 5, 6, [False,False,True,False,False,False]]
        # slu_stage = ['descent',5,6,0.6,.1,[True,True,True,False,False,False],[None,None,None,None,None,None]]
        # sl_stage = ['descent',5,6,0.5,.1,[True,True,False,False,False,False],[None,None,None,None,None,None]]
        # u_sweep_2 = ['sweep', 20, 6, [False,False,True,False,False,False]]
        # s_flip_check_6 = ['flip',6,[True,False,False,False,False,False]]
        # lu_fine_tune = ['descent',10,6,0.4,.01,[True,True,True,False,False,False],[None,None,None,None,None,None]]

        # stages = [s_sweep, l_sweep, s_flip_check_3, sl_rough, s_flip_check_3, u_sweep, slu_stage, sl_stage, u_sweep_2, slu_stage, s_flip_check_6, lu_fine_tune]
        if starting_point is None:
            angles = np.array([0,0.2,1.25,0,0,0], dtype=float)

            # s_sweep = ['sweep', 15, 2, [True,False,False,False,False,False]]
            # l_sweep = ['sweep', 25, 4, [False,True,False,False,False,False]]
            # s_flip_check_3 = ['flip',4,[True,False,False,False,False,False]]
            # sl_rough = ['descent',15,4,0.7,.2,[True,True,False,False,False,False],[0.8,0.5,0.075,0.5,0.5,0.5]]
            # u_sweep = ['sweep', 20, 6, [False,False,True,False,False,False]]
            # slu_stage_1 = ['descent',15,6,0.5,.1,[True,True,True,False,False,False],[None,None,None,None,None,None]]
            # slu_stage_2 = ['descent',15,6,0.5,.05,[True,True,True,False,False,False],[.1,.1,None,None,None,None]]
            # s_flip_check_6 = ['flip',6,[True,False,False,False,False,False]]
            # lu_fine_tune = ['descent',10,6,0.4,.01,[True,True,True,False,False,False],[None,None,None,None,None,None]]

            # stages = [s_sweep, l_sweep, s_flip_check_3, sl_rough, s_flip_check_3, u_sweep, slu_stage_1, slu_stage_2, s_flip_check_6, lu_fine_tune]

            # s_sweep_1 = ['sweep', 20, 2, [True,False,False,False,False,False]]
            # l_sweep_1 = ['sweep', 25, 3, [False,True,False,False,False,False]]
            # s_flip_check_3 = ['flip',4,[True,True,False,False,False,False]]
            # sl_rough = ['descent',15,4,0.7,.2,[True,True,False,False,False,False],[0.8,0.5,0.075,0.5,0.5,0.5]]
            # u_sweep = ['smartsweep', 20, 6, [False,False,True,False,False,False]]
            # slu_stage_1 = ['descent',15,6,0.5,.1,[True,True,True,False,False,False],[None,None,None,None,None,None]]
            # slu_stage_2 = ['descent',15,6,0.5,.05,[True,True,True,False,False,False],[.1,.1,None,None,None,None]]
            # s_flip_check_6 = ['flip',6,[True,False,False,False,False,False]]
            # lu_fine_tune = ['descent',10,6,0.4,.01,[True,True,True,False,False,False],[None,None,None,None,None,None]]

            # stages = [s_sweep_1, l_sweep_1, s_flip_check_3, sl_rough, s_flip_check_3, u_sweep, slu_stage_1, slu_stage_2, s_flip_check_6, lu_fine_tune]


            
            s_sweep_1 = ['sweep', 20, 2, [True,False,False,False,False,False]]
            l_sweep_1 = ['sweep', 25, 3, [False,True,False,False,False,False]]
            s_flip_check_3 = ['flip',4,[True,True,False,False,False,False]]
            sl_rough = ['descent',15,4,0.7,.2,[True,True,False,False,False,False],[0.8,0.5,0.075,0.5,0.5,0.5]]
            u_sweep = ['smartsweep', 20, 6, [False,False,True,False,False,False]]
            slu_stage_1 = ['descent',15,6,0.5,.1,[True,True,True,False,False,False],[None,None,None,None,None,None]]
            slu_stage_2 = ['descent',15,6,0.5,.05,[True,True,True,False,False,False],[.1,.1,None,None,None,None]]
            s_flip_check_6 = ['flip',6,[True,False,False,False,False,False]]
            lu_fine_tune = ['descent',10,6,0.4,.01,[True,True,True,False,False,False],[None,None,None,None,None,None]]

            stages = [s_sweep_1, l_sweep_1, s_flip_check_3, sl_rough, s_flip_check_3, u_sweep, slu_stage_1, slu_stage_2, s_flip_check_6, lu_fine_tune]
        else:
            angles = starting_point

            sl_rough = ['descent',15,4,0.7,.2,[True,True,False,False,False,False],[0.8,0.5,0.075,0.5,0.5,0.5]]
            slu_stage_1 = ['descent',15,6,0.5,.1,[True,True,True,False,False,False],[.1,.1,None,None,None,None]]
            s_flip_check_6 = ['flip',6,[True,False,False,False,False,False]]
            lu_fine_tune = ['descent',10,6,0.4,.01,[True,True,True,False,False,False],[None,None,None,None,None,None]]

            stages = [sl_rough, slu_stage_1, s_flip_check_6, lu_fine_tune]

        self.renderer.setJointAngles(angles)

        self._load_target(segmentation_data, target_depth)

        for stage in stages:

            if stage[0] == 'descent':

                for i in range(6):
                    if stage[6][i] is not None:
                        angle_learning_rate[i] = stage[6][i]

                do_ang = np.array(stage[5])
                self.renderer.setMaxParts(stage[2])

                for i in range(stage[1]):
                    for idx in np.where(do_ang)[0]:

                        if abs(np.mean(history,0)[idx] - angles[idx]) <= angle_learning_rate[idx]:
                            angle_learning_rate[idx] *= stage[3]

                        angle_learning_rate = np.max((angle_learning_rate,self.min_learning_rate),0)

                        # Under
                        temp = angles.copy()
                        temp[idx] -= angle_learning_rate[idx]
                        if temp[idx] >= self.u_reader.joint_limits[idx][0] and temp[idx] <= self.u_reader.joint_limits[idx][1]:
                            self.renderer.setJointAngles(temp)
                            color, depth = self.renderer.render()
                            under_err = self._total_err(stage[2], color, depth)
                            if self.preview:
                                self._show(color, depth, target_depth)
                        else:
                            under_err = np.inf

                        # Over
                        temp = angles.copy()
                        temp[idx] += angle_learning_rate[idx]
                        if temp[idx] >= self.u_reader.joint_limits[idx,0] and temp[idx] <= self.u_reader.joint_limits[idx,1]:
                            self.renderer.setJointAngles(temp)
                            color, depth = self.renderer.render()
                            over_err = self._total_err(stage[2], color, depth)
                            if self.preview:
                                self._show(color, depth, target_depth)
                        else:
                            over_err = np.inf

                        if over_err < under_err:
                            angles[idx] += angle_learning_rate[idx]
                        elif over_err > under_err:
                            angles[idx] -= angle_learning_rate[idx]


                    history[1:] = history[:-1]
                    history[0] = angles

                    err_history[1:] = err_history[:-1]
                    err_history[0] = min(over_err, under_err)
                    if abs(np.mean(err_history) - err_history[0])/err_history[0] < stage[4]:
                        break

                    # Angle not changing
                    if ((history.max(0) - history.min(0) <= self.min_learning_rate) + np.isclose((history.max(0) - history.min(0)),self.min_learning_rate)).all():
                        break
                    if (history[:3] == history[0]).all():
                        break

            elif stage[0] == 'flip':

                do_ang = np.array(stage[2])
                self.renderer.setMaxParts(stage[1])

                for idx in np.where(do_ang)[0]:
                    temp = angles.copy()
                    temp[idx] *= -1
                    if temp[idx] >= self.u_reader.joint_limits[idx,0] and temp[idx] <= self.u_reader.joint_limits[idx,1]:
                        self.renderer.setJointAngles(temp)
                        color, depth = self.renderer.render()
                        err = self._total_err(stage[1], color, depth)

                        if err < err_history[0]:
                            angles[idx] *= -1

                            if self.preview:
                                self.renderer.setJointAngles(angles)
                                color, depth = self.renderer.render()
                                self._show(color, depth, target_depth)

                            history[1:] = history[:-1]
                            history[0] = angles
                            err_history[1:] = err_history[:-1]
                            err_history[0] = err

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
                        space_err.append(self._total_err(stage[2], color, depth))
                        if self.preview:
                            self._show(color, depth, target_depth)

                    angles = space[space_err.index(min(space_err))]

            elif stage[0] == 'smartsweep':
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
                        space_err.append(self._total_err(stage[2], color, depth))
                        if self.preview:
                            self._show(color, depth, target_depth)

                    ang_space = space[:,idx]
                    err_pred = interp1d(ang_space, np.array(space_err), kind='cubic')
                    x = np.linspace(self.u_reader.joint_limits[idx,0], self.u_reader.joint_limits[idx,1], div*5)
                    predicted_errors = err_pred(x)
                    pred_min_ang = x[predicted_errors.argmin()]
                    angles[idx] = pred_min_ang
                    self.renderer.setJointAngles(angles)
                    color, depth = self.renderer.render()
                    if self._total_err(stage[2], color, depth) > min(space_err): 
                        angles = space[space_err.index(min(space_err))]

        return angles



    def _downsample(self, base, factor):
        dims = [x//factor for x in base.shape[0:2]]
        dims.reverse()
        return cv2.resize(base, tuple(dims))

    def _reorganize_by_link(self, data):
        out = {}
        for idx in range(len(data['class_ids'])):
            id = data['class_ids'][idx]
            out[self.classes[id]] = {
                'roi':data['rois'][idx],
                'confidence':data['scores'][idx],
                'mask':data['masks'][...,idx]
                }
        return out

    def _mask_err(self, num_joints, render):
        color_dict = self.renderer.getColorDict()
        err = 0
        count = 0
        for link in self.link_names[:num_joints]:
            if link in self._target_masks.keys():
                count += 1
                joint_mask = self._target_masks[link]
                render_mask = np.all(render == color_dict[link], axis=-1)

                diff = joint_mask != render_mask
                err += np.sum(diff) / np.sum(joint_mask)

        return (err / count) / 5

    
    def _depth_err(self, num_joints, render, render_color):

        color_dict = self.renderer.getColorDict()
        err = 0

        # Matched Error
        for link in self.link_names[:num_joints]:
            if link in self._masked_targets.keys():
                target_masked = self._masked_targets[link]

                render_mask = np.all(render_color == color_dict[link], axis=-1)
                render_masked = render * render_mask

                diff = target_masked - render_masked
                diff = np.abs(diff) ** .5
                err += np.mean(diff[diff!=0]) #+ np.median(diff[diff!=0])

        # Unmatched Error
        # link_mask = self._masked_targets[]
        # for link in self.link_names[:num_joints]:
        #     if link in self._masked_targets.keys():
        #         joint_masks = joint_masks + seg_data[self.link_names[idx]]['mask']

        diff = self._tgt_depth - render
        diff = np.abs(diff) ** 0.5
        err += np.mean(diff[diff!=0])
        
        return err


    def _load_target(self, seg_data, tgt_depth):
        self._masked_targets = {}
        self._target_masks = {}
        self._tgt_depth = tgt_depth
        for link in self.link_names:
            if link in seg_data.keys():
                link_mask = seg_data[self.link_names[self.link_names.index(link)]]['mask']
                target_masked = link_mask * tgt_depth
                self._masked_targets[link] = target_masked
                self._target_masks[link] = link_mask


    def _total_err(self, num_joints, render_color, render_depth):
        d = self._depth_err(num_joints,render_depth,render_color)
        #m = self._mask_err(num_joints,render_color)
        #print(f"Depth: {(d*100)/(d+m):.1f}%\tMask: {(m*100)/(d+m):.1f}%")
        return d #+ m

    def _show(self, color, depth, target_depth):
        size = color.shape[0:2]
        dim = [x*2 for x in size]
        dim.reverse()
        dim = tuple(dim)
        color = cv2.resize(color, dim, interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, dim, interpolation=cv2.INTER_NEAREST)
        target_depth = cv2.resize(target_depth, dim)
        cv2.imshow("Color",color)
        d = color_array(target_depth - depth)
        
        cv2.imshow("Depth",d)
        cv2.waitKey(10)