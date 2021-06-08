# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import cv2

import numpy as np
import tensorflow as tf
from pixellib.instance import custom_segmentation
from scipy.interpolate import interp1d

from ..simulation.lookup import RobotLookupManager
from ..simulation.render import Renderer
from ..turbo_colormap import color_array
from ..urdf import URDFReader
from ..utils import str_to_arr, get_gpu_memory

tf.compat.v1.enable_eager_execution()

# DEFAULT_CAMERA_POSE = [.042,-1.425,.75, -.01,1.553,-.057]
DEFAULT_CAMERA_POSE = [0, -1.5, .75, 0, np.pi/2, 0]

class CameraPredictor():
    def __init__(self,
        base_pose = DEFAULT_CAMERA_POSE,
        ds_factor: int = 8,
        preview: bool = False,
        save_to: str = None,
        min_angle_inc = np.array([.005]*6),
        history_length = 5
        ):

        self.base_pose = np.array(base_pose)

        self.ds_factor = ds_factor
        self.preview = preview
        if preview:
            self.viz = ProjectionViz(save_to)
        self.min_ang_inc = min_angle_inc
        self.history_length = history_length

        self.intrinsics = f'1280_720_color_{self.ds_factor}'

        self.u_reader = URDFReader()
        self.renderer = Renderer('seg',None,self.intrinsics)

        self.classes = ["BG"]
        self.classes.extend(self.u_reader.mesh_names[:6])
        self.link_names = self.classes[1:]

        self.seg = custom_segmentation()
        self.seg.inferConfig(num_classes=6, class_names=self.classes)
        self.seg.load_model("models/segmentation/multi/B.h5")

        self.renderer.setMaxParts(None)


    # def changeCameraPose(self, camera_pose):
    #     self.camera_pose = camera_pose
    #     self.renderer.setCameraPose(camera_pose)
    #     self._loadLookup()


    # def _loadLookup(self):
    #     max_elements = int(get_gpu_memory()[0] / (3 * 32))

    #     lm = RobotLookupManager()
    #     ang, depth = lm.get(self.intrinsics, self.camera_pose, 4,
    #         np.array([True,True,False,False,False,False]), max_elements)

    #     self.lookup_angles = ang
    #     self.lookup_depth = tf.pow(tf.constant(depth,tf.float32),0.5)



    def _setStages(self):

        # Stages in form:
        # Sweep/Smartsweep:
        #   Divisions, offset to render, angles_to_edit
        # Descent: 
        #   Iterations, rate reduction, early_stop_thresh, angles_to_edit, inital_learning_rate

        
        coarse_descent = ['descent', 50, 0.5, .1, [True]*6, [0.25,0.25,0.25,0.05,0.05,0.05]]
        wide_tensorsweep_xyz = ['tensorsweep', 50, .3, [True,True,True,False,False,False]]
        wide_tensorsweep_rpy = ['tensorsweep', 50, .1, [False,False,False,True,True,True]]
        # u_sweep_narrow = ['smartsweep', 10, .1, [False,False,True,False,False,False]]
        
        # rb_fine_tune = ['descent', 5, 0.4, .015, [False,False,False,True,True,False], [None,None,None,.005,.005,None]]
        # full_tune = ['descent', 10, 0.4, .015, [True,True,True,True,True,False], [None,None,None,None,None,None]]
        
        self.stages = [coarse_descent, *([wide_tensorsweep_xyz, wide_tensorsweep_rpy]*2)]



    def run(self, og_image, target_depth, robot_pose):
        self.renderer.setJointAngles(robot_pose)

        if self.preview:
            self.viz.loadTargetColor(og_image)
            self.viz.loadTargetDepth(target_depth)

        target_depth = self._downsample(target_depth, self.ds_factor)
        r, output = self.seg.segmentImage(self._downsample(og_image, self.ds_factor), process_frame=True)
        segmentation_data = self._reorganize_by_link(r)

        if self.preview:
            self.viz.loadSegmentedLinks(output)

        learning_rates = np.zeros(6)

        history = np.zeros((self.history_length, 6))
        err_history = np.zeros(self.history_length)
        pose = np.copy(self.base_pose)

        self._setStages()

        self._load_target(segmentation_data, target_depth)

        def preview_if_applicable(color, depth):
            if self.preview:
                self.viz.loadRenderedColor(color)
                self.viz.loadRenderedDepth(depth)
                self.viz.show()

        def render_at_pose(pose):
            self.renderer.setCameraPose(pose)
            return self.renderer.render()

        for stage in self.stages:

            # if stage[0] == 'lookup':

            #     diff = self._tgt_depth_stack_half - self.lookup_depth
            #     diff = tf.abs(diff) 
            #     lookup_err = tf.reduce_mean(diff, (1,2)) *- tf.math.reduce_std(diff, (1,2))

            #     pose = self.lookup_angles[tf.argmin(lookup_err).numpy()]
            # elif stage[0] == 'descent':
            if stage[0] == 'descent':

                for i in range(6):
                    if stage[5][i] is not None:
                        learning_rates[i] = stage[5][i]

                do_param = np.array(stage[4])

                for i in range(stage[1]):
                    for idx in np.where(do_param)[0]:

                        if abs(np.mean(history,0)[idx] - pose[idx]) <= learning_rates[idx]:
                            learning_rates[idx] *= stage[2]

                        learning_rates = np.max((learning_rates,self.min_ang_inc),0)

                        def in_limits(ang):
                            return ang >= self.u_reader.joint_limits[idx][0] and ang <= self.u_reader.joint_limits[idx][1]

                        # Under
                        temp = pose.copy()
                        temp[idx] -= learning_rates[idx]
                        if in_limits(temp[idx]):
                            color, depth = render_at_pose(temp)
                            under_err = self._error(color, depth)
                            preview_if_applicable(color, depth)

                        else:
                            under_err = np.inf

                        # Over
                        temp[idx] += 2 * learning_rates[idx]
                        if in_limits(temp[idx]):
                            color, depth = render_at_pose(temp)
                            over_err = self._error(color, depth)
                            preview_if_applicable(color, depth)
                        else:
                            over_err = np.inf

                        if over_err < under_err:
                            pose[idx] += learning_rates[idx]
                        elif over_err > under_err:
                            pose[idx] -= learning_rates[idx]


                    history[1:] = history[:-1]
                    history[0] = pose

                    err_history[1:] = err_history[:-1]
                    err_history[0] = min(over_err, under_err)
                    if abs(np.mean(err_history) - err_history[0])/err_history[0] < stage[3]:
                        break

                    # Angle not changing
                    if ((history.max(0) - history.min(0) <= self.min_ang_inc) + np.isclose((history.max(0) - history.min(0)),self.min_ang_inc)).all():
                        break
                    if (history[:3] == history[0]).all():
                        break

            # elif stage[0] == 'flip':

            #     do_param = np.array(stage[2])

            #     for idx in np.where(do_param)[0]:
            #         temp = pose.copy()
            #         temp[idx] *= -1
            #         if temp[idx] >= self.u_reader.joint_limits[idx,0] and temp[idx] <= self.u_reader.joint_limits[idx,1]:
            #             self.renderer.setJointAngles(temp)
            #             color, depth = self.renderer.render()
            #             err = self._error(color, depth)

            #             if err < err_history[0]:
            #                 pose[idx] *= -1

            #                 if self.preview:
            #                     color, depth = render_at_pose(pose)
            #                     preview_if_applicable(color, depth)

            #                 history[1:] = history[:-1]
            #                 history[0] = pose
            #                 err_history[1:] = err_history[:-1]
            #                 err_history[0] = err

            elif stage[0] == 'smartsweep':

                do_param = np.array(stage[3])
                div = stage[1]

                color, depth = render_at_pose(pose)
                base_err = self._error(color, depth)

                for idx in np.where(do_param)[0]:
                    temp_low = pose.copy()
                    temp_high = pose.copy()

                    temp_low[idx] = temp_low[idx]-stage[2]
                    temp_high[idx] = temp_low[idx]+stage[2]

                    space = np.linspace(temp_low, temp_high, div)
                    space_err = []
                    for angs in space:
                        color, depth = render_at_pose(angs)
                        space_err.append(self._error(color, depth))
                        preview_if_applicable(color, depth)

                    ang_space = space[:,idx]
                    err_pred = interp1d(ang_space, np.array(space_err), kind='cubic')
                    x = np.linspace(temp_low[idx], temp_high[idx], div*5)
                    predicted_errors = err_pred(x)
                    pred_min_ang = x[predicted_errors.argmin()]

                    angs = pose.copy()
                    angs[idx] = pred_min_ang
                    color, depth = render_at_pose(angs)
                    pred_min_err = self._error(color, depth)

                    errs = [base_err, min(space_err), pred_min_err]
                    min_type = errs.index(min(errs))
                    
                    if min_type == 1: 
                        pose = space[space_err.index(min(space_err))]
                        err_history[1:] = err_history[:-1]
                        err_history[0] = min(space_err)

                    elif min_type == 2:
                        pose = angs
                        err_history[1:] = err_history[:-1]
                        err_history[0] = pred_min_err

                    history[1:] = history[:-1]
                    history[0] = pose

                    if self.preview:
                        color, depth = render_at_pose(pose)
                        preview_if_applicable(color, depth)

            elif stage[0] == 'tensorsweep':

                do_param = np.array(stage[3])
                div = stage[1]

                for idx in np.where(do_param)[0]:
                    temp_low = pose.copy()
                    temp_high = pose.copy()

                    temp_low[idx] = temp_low[idx]-stage[2]
                    temp_high[idx] = temp_low[idx]+stage[2]

                    space = np.linspace(temp_low, temp_high, div)
                    depths = np.zeros((div, *self.renderer.resolution))
                    for angs, i in zip(space, range(div)):
                        color, depth = render_at_pose(angs)
                        depths[i] = depth
                        preview_if_applicable(color, depth)

                    lookup_depth = tf.pow(tf.constant(depths,tf.float32),0.5)
                    stack = tf.stack([tf.pow(tf.constant(self._tgt_depth, tf.float32),0.5)]*div)

                    diff = tf.abs(stack - lookup_depth)
                    lookup_err = tf.reduce_mean(diff, (1,2)) *- tf.math.reduce_std(diff, (1,2))

                    pose = space[tf.argmin(lookup_err).numpy()]

                    if self.preview:
                        color, depth = render_at_pose(pose)
                        preview_if_applicable(color, depth)

        return pose


    def _downsample(self, base: np.ndarray, factor: int) -> np.ndarray:
        dims = [x//factor for x in base.shape[0:2]]
        dims.reverse()
        return cv2.resize(base, tuple(dims))

    def _reorganize_by_link(self, data: dict) -> dict:
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
                out[self.classes[id]]['roi'] = [ #TODO: Test this functionality
                    np.min([out[self.classes[id]]['roi'][:2],data['rois'][idx][:2]]),
                    np.max([out[self.classes[id]]['roi'][2:],data['rois'][idx][2:]])]
        return out

    def _load_target(self, seg_data: dict, tgt_depth: np.ndarray) -> None:
        self._masked_targets = {}
        self._target_masks = {}
        self._tgt_depth = tgt_depth
        # self._tgt_depth_stack_half = tf.stack([tf.pow(tf.constant(tgt_depth, tf.float32),0.5)]*len(self.lookup_angles))
        for link in self.link_names:
            if link in seg_data.keys():
                link_mask = seg_data[self.link_names[self.link_names.index(link)]]['mask']
                target_masked = link_mask * tgt_depth
                self._masked_targets[link] = target_masked
                self._target_masks[link] = link_mask
                
    def _error(self, render_color: np.ndarray, render_depth: np.ndarray) -> float:
        color_dict = self.renderer.color_dict
        err = 0

        # Matched Error
        for link in self.link_names:
            if link in self._masked_targets.keys():
                target_masked = self._masked_targets[link]
                joint_mask = self._target_masks[link]

                # NOTE: Instead of matching color, this matches blue values,
                #       as each of the default colors has a unique blue value when made.
                render_mask = render_color[...,0] == color_dict[link][0]
                render_masked = render_depth * render_mask

                # Mask
                diff = joint_mask != render_mask
                err += np.mean(diff)

                # Only do if enough depth data present (>5% of required pixels have depth data)
                if np.sum(target_masked != 0) > (.05 * np.sum(joint_mask)):
                    # Depth
                    diff = target_masked - render_masked
                    diff = np.abs(diff) ** .5
                    if diff[diff!=0].size > 0:
                        err += np.mean(diff[diff!=0])

        # Unmatched Error
        diff = self._tgt_depth - render_depth
        diff = np.abs(diff) ** 0.5
        #err += np.mean(diff[diff!=0])
        err += np.mean(diff[diff!=0]) *- np.std(diff[diff!=0])

        return err



class ProjectionViz():

    def __init__(self, video_path = None, fps = 45, resolution = (1280, 720)) -> None:
        self.write_to_file = video_path is not None
        self.resolution = resolution
        if video_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(video_path, fourcc, fps, resolution)

        self.res = np.flip(np.array(self.resolution))
        self.resize_to = tuple(np.array(self.resolution) // 2)
        self.frame = np.zeros((*self.res,3), dtype=np.uint8)

        self.input_side_up_to_date = False

    def loadTargetColor(self, target_color: np.ndarray) -> None:
        self.tgt_color = target_color
        self.input_side_up_to_date = False

    def loadTargetDepth(self, target_depth: np.ndarray) -> None:
        self.tgt_depth = target_depth
        self.input_side_up_to_date = False

    def loadSegmentedLinks(self, segmented_color: np.ndarray) -> None:
        self.seg_links = segmented_color
        self.input_side_up_to_date = False

    def loadRenderedColor(self, render_color: np.ndarray) -> None: self.rend_color = render_color

    def loadRenderedDepth(self, render_depth: np.ndarray) -> None: self.rend_depth = render_depth

    def _genInput(self):
        self.frame[:self.res[0]//2, :self.res[1]//2] = self._orig()
        self.frame[self.res[0]//2:, :self.res[1]//2] = self._seg()

        # Add overlay
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.frame = cv2.putText(self.frame, "Input Color/Depth", (10,30), font, 1, color, 2, cv2.LINE_AA, False)
        self.frame = cv2.putText(self.frame, "Detected Links", (10,self.res[0]//2 + 30), font, 1, color, 2, cv2.LINE_AA, False)
        self.input_side_up_to_date = True


    def show(self) -> None:
        if not self.input_side_up_to_date:
            self._genInput()
        self.frame[:self.res[0]//2, self.res[1]//2:] = cv2.resize(self.rend_color, self.resize_to)
        self.frame[self.res[0]//2:, self.res[1]//2:] = self._depth()

        # Add overlay
        color = (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        self.frame = cv2.line(self.frame, (0,self.res[0]//2), (self.res[1],self.res[0]//2), color, thickness=3)
        self.frame = cv2.line(self.frame, (self.res[1]//2,0), (self.res[1]//2,self.res[0]), color, thickness=3)
        self.frame = cv2.putText(self.frame, "Render", (self.res[1]//2 + 10, 30), font, 1, color, 2, cv2.LINE_AA, False)
        self.frame = cv2.putText(self.frame, "Render Depth vs. Input Depth", (self.res[1]//2 + 10,self.res[0]//2 + 30), font, 1, color, 2, cv2.LINE_AA, False) 

        cv2.imshow("Projection Matcher", self.frame)
        cv2.waitKey(1)
        if self.write_to_file:
            self.writer.write(self.frame)

    def _seg(self):
        return  cv2.resize(self.seg_links, self.resize_to)

    def _orig(self):
        COLOR_ALPHA = .6
        color = cv2.resize(self.tgt_color, self.resize_to)
        depth = color_array(cv2.resize(self.tgt_depth, self.resize_to), percent=5)
        return cv2.addWeighted(color, COLOR_ALPHA, depth, (1-COLOR_ALPHA), 0)

    def _depth(self):
        tgt_d = cv2.resize(self.tgt_depth, self.resize_to, interpolation=cv2.INTER_NEAREST)
        d = cv2.resize(self.rend_depth, self.resize_to, interpolation=cv2.INTER_NEAREST)
        return color_array(tgt_d - d)

    def __del__(self) -> None:
        if self.write_to_file:
            self.writer.release()