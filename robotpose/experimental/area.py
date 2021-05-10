# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from robotpose.data.dataset import Dataset
import numpy as np
import h5py

import cv2
from pixellib.instance import custom_segmentation
from scipy.interpolate import interp1d

from robotpose.simulation.render import BaseRenderer, SkeletonRenderer
from robotpose.urdf import URDFReader
from ..turbo_colormap import color_array

from tqdm import tqdm


CAMERA_POSE = [.042,-1.425,.399, -.01,1.553,-.057]
WIDTH = 800


LOOKUP_LOCATION = 'lookups'

class LookupCreator(SkeletonRenderer):
    def __init__(self, camera_pose, ds_factor = 8):

        self.u_reader = URDFReader()
        super().__init__('BASE', 'seg', camera_pose=camera_pose, camera_intrin=f'1280_720_color_{ds_factor}')

    def load_config(self, joints_to_render, angles_to_do, divisions):
        self.setMaxParts(joints_to_render)
        divisions = np.array(divisions)
        angles_to_do = np.array(angles_to_do)

        divisions[~angles_to_do] = 1
        self.num = np.prod(divisions)

        self.angles = np.zeros((self.num,6))

        for idx in np.where(angles_to_do)[0]:
            angle_range = np.linspace(self.u_reader.joint_limits[idx,0],self.u_reader.joint_limits[idx,1],divisions[idx])

            repeat = np.prod(divisions[:idx])
            tile = self.num//(repeat*divisions[idx])

            self.angles[:,idx] = np.tile(np.repeat(angle_range,repeat),tile)

    def run(self, file_name, preview = True):

        self.setJointAngles([0,0,0,0,0,0])
        color, depth = self.render()

        color_arr = np.zeros((self.num, *color.shape), dtype=np.uint8)
        depth_arr = np.zeros((self.num, *color.shape[:2]), dtype=float)

        for pose,idx in tqdm(zip(self.angles, range(len(self.angles))),total=len(self.angles),desc="Rendering Lookup Table"):
            self.setJointAngles(pose)
            color, depth = self.render()
            color_arr[idx] = color
            depth_arr[idx] = depth
            if preview:
                self._show(color)

        with tqdm(total=3, desc=f"Writing to {file_name}") as pbar:
            f = h5py.File(file_name, 'w')
            f.create_dataset('angles',data=self.angles)
            pbar.update(1)
            f.create_dataset('color',data=color_arr, compression="gzip", compression_opts=1)
            pbar.update(1)
            f.create_dataset('depth',data=depth_arr, compression="gzip", compression_opts=1)
            pbar.update(1)


    def _show(self, color):
        size = color.shape[0:2]
        dim = [x*8 for x in size]
        dim.reverse()
        dim = tuple(dim)
        color = cv2.resize(color, dim, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Lookup Table Creation",color)
        cv2.waitKey(1)




class ProjectionMatcher():
    def __init__(self, camera_pose = CAMERA_POSE, ds_factor = 8, preview = False):
        self.camera_pose = camera_pose
        self.ds_factor = ds_factor
        self.do_angle = np.array([True,True,True,False,False,False])
        self.min_learning_rate = np.array([.005]*6)
        self.history_length = 5
        self.preview = preview

        self.u_reader = URDFReader()
        self.renderer = SkeletonRenderer('BASE','seg',camera_pose,f'1280_720_color_{self.ds_factor}')

        self.classes = ["BG"]
        self.classes.extend(self.u_reader.mesh_names[:6])
        self.link_names = self.classes[1:]

        self.seg = custom_segmentation()
        self.seg.inferConfig(num_classes=6, class_names=self.classes)
        self.seg.load_model("models/segmentation/multi/B.h5")


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
        # Sweep/Smartsweep:
        #   Divisions, joints to render, offset to render, angles to edit
        # Descent: 
        #   Iterations, joints to render, rate reduction, early stop thresh, angles to edit, inital learning rate
        # Flip: 
        #   joints to render, edit_angles
        # Checkmod: 
        #   joints to render, offset1, multiplier, offset2, edit_angles

        if starting_point is None:
            angles = np.array([0,0.2,1.25,0,0,0], dtype=float)

            s_sweep_1 = ['smartsweep', 15, 2, None, [True,False,False,False,False,False]]
            l_sweep_1 = ['smartsweep', 20, 4, None, [False,True,False,False,False,False]]
            s_flip_check_3 = ['flip',4,[True,True,False,False,False,False]]
            sl_rough = ['descent',15,4,0.7,.5,[True,True,False,False,False,False],[0.75,0.5,0.4,0.5,0.5,0.5]]
            u_sweep = ['smartsweep', 10, 6, None, [False,False,True,False,False,False]]
            slu_stage_1 = ['descent',15,6,0.5,.2,[True,True,True,False,False,False],[None,None,None,None,None,None]]
            slu_stage_2 = ['descent',15,6,0.5,.1,[True,True,True,False,False,False],[.1,.1,None,None,None,None]]
            s_flip_check_6 = ['flip', 6, [True,False,False,False,False,False]]
            sl_sweep_check = ['smartsweep', 10, 6, .25, [True,True,False,False,False,False]]
            u_sweep_check = ['smartsweep', 25, 6, None, [False,False,True,False,False,False]]
            lu_fine_tune = ['descent',10,6,0.4,.015,[True,True,True,False,False,False],[None,None,None,None,None,None]]
            

            stages = [s_sweep_1, l_sweep_1, s_flip_check_3, sl_rough, s_flip_check_3, u_sweep, slu_stage_1, slu_stage_2, s_flip_check_6, sl_sweep_check, u_sweep_check, lu_fine_tune]
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

                do_ang = np.array(stage[4])
                self.renderer.setMaxParts(stage[2])
                div = stage[1]

                self.renderer.setJointAngles(angles)
                color, depth = self.renderer.render()
                base_err = self._total_err(stage[2], color, depth)

                for idx in np.where(do_ang)[0]:
                    temp_low = angles.copy()
                    temp_high = angles.copy()
                    if stage[3] is None:
                        temp_low[idx] = self.u_reader.joint_limits[idx,0]      
                        temp_high[idx] = self.u_reader.joint_limits[idx,1]
                    else:
                        temp_low[idx] = max(temp_low[idx]-stage[3], self.u_reader.joint_limits[idx,0])
                        temp_high[idx] = min(temp_low[idx]+stage[3], self.u_reader.joint_limits[idx,1])

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
                    x = np.linspace(temp_low[idx], temp_high[idx], div*5)
                    predicted_errors = err_pred(x)
                    pred_min_ang = x[predicted_errors.argmin()]

                    angs = angles.copy()
                    angs[idx] = pred_min_ang
                    self.renderer.setJointAngles(angs)
                    color, depth = self.renderer.render()
                    pred_min_err = self._total_err(stage[2], color, depth)

                    errs = [base_err, min(space_err), pred_min_err]
                    min_type = errs.index(min(errs))
                    
                    if min_type == 1: 
                        angles = space[space_err.index(min(space_err))]
                    elif min_type == 2:
                        angles = angs

                    if self.preview:
                        self.renderer.setJointAngles(angles)
                        color, depth = self.renderer.render()
                        self._show(color, depth, target_depth)

        return angles



    def _downsample(self, base, factor):
        dims = [x//factor for x in base.shape[0:2]]
        dims.reverse()
        return cv2.resize(base, tuple(dims))

    def _reorganize_by_link(self, data):
        out = {}
        for idx in range(len(data['class_ids'])):
            id = data['class_ids'][idx]
            if id not in data['class_ids'][:idx]:
                out[self.classes[id]] = {
                    'roi':data['rois'][idx],
                    'confidence':data['scores'][idx],
                    'mask':data['masks'][...,idx]
                    }
            else:
                out[self.classes[id]]['mask'] += data['masks'][...,idx]
                out[self.classes[id]]['confidence'] = max(out[self.classes[id]]['confidence'], data['scores'][idx])
                #out[self.classes[id]]['rois'] = [np.min([out[self.classes[id]]['rois'][:2],data['rois'][idx][:2]],0)]
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
                err += np.mean(diff)

        return err

    
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
                if diff[diff!=0].size > 0:
                    err += np.mean(diff[diff!=0])


        # Unmatched Error
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
        m = self._mask_err(num_joints,render_color)
        #print(f"Depth: {(d*100)/(d+m):.1f}% ({d:.2f})\tMask: {(m*100)/(d+m):.1f}% ({m:.2f})")
        return d + m

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
        cv2.waitKey(1)









class ProjectionMatcherLookup():
    def __init__(self, camera_pose = CAMERA_POSE, ds_factor = 8, preview = False):
        self.camera_pose = camera_pose
        self.ds_factor = ds_factor
        self.do_angle = np.array([True,True,True,False,False,False])
        self.min_learning_rate = np.array([.005]*6)
        self.history_length = 5
        self.preview = preview

        self.u_reader = URDFReader()
        self.renderer = SkeletonRenderer('BASE','seg',camera_pose,f'1280_720_color_{self.ds_factor}')

        self.classes = ["BG"]
        self.classes.extend(self.u_reader.mesh_names[:6])
        self.link_names = self.classes[1:]

        self.seg = custom_segmentation()
        self.seg.inferConfig(num_classes=6, class_names=self.classes)
        self.seg.load_model("models/segmentation/multi/B.h5")

        with h5py.File('test1.h5','r') as f:
            self.lookup_angles = np.copy(f['angles'])
            self.lookup_depth = np.copy(f['depth'])


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
        # Sweep/Smartsweep:
        #   Divisions, joints to render, offset to render, angles to edit
        # Descent: 
        #   Iterations, joints to render, rate reduction, early stop thresh, angles to edit, inital learning rate
        # Flip: 
        #   joints to render, edit_angles
        # Checkmod: 
        #   joints to render, offset1, multiplier, offset2, edit_angles

        if starting_point is None:
            angles = np.array([0,0.2,1.25,0,0,0], dtype=float)

            # lookup = ['lookup', 4]
            # u_sweep = ['smartsweep', 25, 6, None, [False,False,True,False,False,False]]
            # u_stage = ['descent',30,6,0.5,.1,[False,False,True,False,False,False],[0.1,0.1,0.4,0.5,0.5,0.5]]
            # s_flip_check_6 = ['flip', 6, [True,False,False,False,False,False]]
            # sl_sweep_check = ['smartsweep', 5, 6, .25, [True,True,False,False,False,False]]
            # lu_fine_tune = ['descent',10,6,0.4,.015,[True,True,True,False,False,False],[None,None,None,None,None,None]]
            
            # stages = [lookup, u_sweep, u_stage, s_flip_check_6, sl_sweep_check, lu_fine_tune]

            lookup = ['lookup', 6]
            #slu_sweep_check = ['smartsweep', 7, 4, .15, [True,True,True,False,False,False]]
            slu_fine_tune = ['descent',15,6,0.4,.015,[True,True,True,False,False,False],[.05,.05,.05,None,None,None]]
            s_flip_check_6 = ['flip', 6, [True,False,False,False,False,False]]
            
            stages = [lookup, slu_fine_tune, s_flip_check_6]
        else:
            angles = starting_point

            area_sweeps = ['smartsweep', 10, 6, .4, [True,True,True,False,False,False]]
            slu_stage = ['descent',30,6,0.5,.1,[False,False,True,False,False,False],[0.1,0.1,0.1,0.5,0.5,0.5]]
            s_flip_check_6 = ['flip', 6, [True,False,False,False,False,False]]
            lu_fine_tune = ['descent',10,6,0.4,.015,[True,True,True,False,False,False],[None,None,None,None,None,None]]
            
            stages = [area_sweeps, slu_stage, s_flip_check_6, lu_fine_tune]

        self.renderer.setJointAngles(angles)

        self._load_target(segmentation_data, target_depth)

        for stage in stages:

            if stage[0] == 'lookup':

                diff = self._tgt_depth_stack - self.lookup_depth
                diff = np.abs(diff) ** 0.5
                lookup_err = np.mean(diff, (1,2))
                angles = self.lookup_angles[lookup_err.argmin()]

            elif stage[0] == 'descent':

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

                do_ang = np.array(stage[4])
                self.renderer.setMaxParts(stage[2])
                div = stage[1]

                self.renderer.setJointAngles(angles)
                color, depth = self.renderer.render()
                base_err = self._total_err(stage[2], color, depth)

                for idx in np.where(do_ang)[0]:
                    temp_low = angles.copy()
                    temp_high = angles.copy()
                    if stage[3] is None:
                        temp_low[idx] = self.u_reader.joint_limits[idx,0]      
                        temp_high[idx] = self.u_reader.joint_limits[idx,1]
                    else:
                        temp_low[idx] = max(temp_low[idx]-stage[3], self.u_reader.joint_limits[idx,0])
                        temp_high[idx] = min(temp_low[idx]+stage[3], self.u_reader.joint_limits[idx,1])

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
                    x = np.linspace(temp_low[idx], temp_high[idx], div*5)
                    predicted_errors = err_pred(x)
                    pred_min_ang = x[predicted_errors.argmin()]

                    angs = angles.copy()
                    angs[idx] = pred_min_ang
                    self.renderer.setJointAngles(angs)
                    color, depth = self.renderer.render()
                    pred_min_err = self._total_err(stage[2], color, depth)

                    errs = [base_err, min(space_err), pred_min_err]
                    min_type = errs.index(min(errs))
                    
                    if min_type == 1: 
                        angles = space[space_err.index(min(space_err))]
                    elif min_type == 2:
                        angles = angs

                    if self.preview:
                        self.renderer.setJointAngles(angles)
                        color, depth = self.renderer.render()
                        self._show(color, depth, target_depth)

        return angles



    def _downsample(self, base, factor):
        dims = [x//factor for x in base.shape[0:2]]
        dims.reverse()
        return cv2.resize(base, tuple(dims))

    def _reorganize_by_link(self, data):
        out = {}
        for idx in range(len(data['class_ids'])):
            id = data['class_ids'][idx]
            if id not in data['class_ids'][:idx]:
                out[self.classes[id]] = {
                    'roi':data['rois'][idx],
                    'confidence':data['scores'][idx],
                    'mask':data['masks'][...,idx]
                    }
            else:
                out[self.classes[id]]['mask'] += data['masks'][...,idx]
                out[self.classes[id]]['confidence'] = max(out[self.classes[id]]['confidence'], data['scores'][idx])
                #out[self.classes[id]]['rois'] = [np.min([out[self.classes[id]]['rois'][:2],data['rois'][idx][:2]],0)]
        return out



    def _load_target(self, seg_data, tgt_depth):
        self._masked_targets = {}
        self._target_masks = {}
        self._tgt_depth = tgt_depth
        self._tgt_depth_stack = np.stack([tgt_depth]*len(self.lookup_angles))
        for link in self.link_names:
            if link in seg_data.keys():
                link_mask = seg_data[self.link_names[self.link_names.index(link)]]['mask']
                target_masked = link_mask * tgt_depth
                self._masked_targets[link] = target_masked
                self._target_masks[link] = link_mask


    def _total_err(self, num_joints, render_color, render_depth):

        color_dict = self.renderer.getColorDict()
        err = 0

        # Matched Error
        for link in self.link_names[:num_joints]:
            if link in self._masked_targets.keys():
                target_masked = self._masked_targets[link]
                joint_mask = self._target_masks[link]

                render_mask = np.all(render_color == color_dict[link], axis=-1)
                render_masked = render_depth * render_mask

                # Mask
                diff = joint_mask != render_mask
                err += np.mean(diff)

                # Depth
                diff = target_masked - render_masked
                diff = np.abs(diff) ** .5
                if diff[diff!=0].size > 0:
                    err += np.mean(diff[diff!=0])

        # Unmatched Error
        diff = self._tgt_depth - render_depth
        diff = np.abs(diff) ** 0.5
        err += np.mean(diff[diff!=0])

        return err



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
        cv2.waitKey(1)