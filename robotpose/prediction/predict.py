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

from ..constants import DEFAULT_CAMERA_POSE, VIDEO_FPS
from ..crop import Crop, applyBatchCrop, applyCrop
from ..projection import Intrinsics
from ..simulation.lookup import RobotLookupManager
from ..simulation.render import Renderer
from ..training.models import ModelManager
from ..urdf import URDFReader
from ..utils import str_to_arr, color_array

tf.compat.v1.enable_eager_execution()


LOOKUP_NUM_RENDERED = 4
HISTORY_LENGTH = 5

class Predictor():
    def __init__(self,
        camera_pose: np.ndarray = DEFAULT_CAMERA_POSE,
        ds_factor: int = 8,
        preview: bool = False,
        save_to: str = None,
        do_angles: str = 'SLU',
        min_angle_inc: np.ndarray = np.array([.005]*6),
        base_intrin: str = "1280_720_color",
        model_ds: str = 'set10',
        color_dict : dict = None
        ):
        """[summary]

        Parameters
        ----------
        camera_pose : np.ndarray, optional
            [description], by default DEFAULT_CAMERA_POSE
        ds_factor : int, optional
            Factor to downscale images by, by default 8
        preview : bool, optional
            Show a live feed of predictions, by default False
        save_to : str, optional
            File to save video to if previewing, by default None
        do_angles : str, optional
            Angles to predict, by default 'SLU'
        min_angle_inc : np.ndarray, optional
            Smallest change in angle allowed for each joint, by default np.array([.005]*6)
        base_intrin : str, optional
            Intirnsics to use before downscaling, by default "1280_720_color"
        model_ds : str, optional
            Dataset model to use for prediction, by default 'set10'
        color_dict : dict, optional
            Renderer color dictionary. If present, synthetic data is assumed., by default None
        """

        self.ds_factor, self.preview = ds_factor, preview
        if preview:
            self.viz = ProjectionViz(save_to)

        self.do_angles = do_angles.upper()
        self.min_ang_inc, self.history_length = min_angle_inc, HISTORY_LENGTH

        # Set up rendering tools
        self.intrinsics = Intrinsics(base_intrin)
        self.intrinsics.downscale(ds_factor)
        self.u_reader = URDFReader()
        self.renderer = Renderer('seg',camera_pose,self.intrinsics)

        self.synthetic = color_dict is not None
        # Segmentation classes
        self.classes = ["BG"]
        self.classes.extend(self.u_reader.mesh_names[:6])
        self.link_names = self.classes[1:]

        if self.synthetic:
            self.color_dict = color_dict
        else:
            # Set up segmenter
            mm = ModelManager()
            self.seg = custom_segmentation()
            self.seg.inferConfig(num_classes=6, class_names=self.classes)
            self.seg.load_model(mm.dynamicLoad(dataset=model_ds))

        self.crops = Crop(camera_pose, self.intrinsics)

        self.changeCameraPose(camera_pose)  # Set to default camera pose


    def changeCameraPose(self, camera_pose):
        self.camera_pose = camera_pose
        self.renderer.setCameraPose(camera_pose)
        self._loadLookup()


    def _loadLookup(self):

        lm = RobotLookupManager()
        ang, depth, = lm.get(self.intrinsics, self.camera_pose, LOOKUP_NUM_RENDERED, 'SL')

        self.lookup_angles = ang
        self.lookup_depth = tf.pow(tf.constant(depth,tf.float32),0.5)



    def _setStages(self):

        # Stages in form:
        # Lookup:
        #   Num_link_to_render
        # Sweep/Smartsweep:
        #   Divisions, Num_link_to_render, offset to render, angles_to_edit
        # Descent:
        #   Iterations, Num_link_to_render, rate reduction, early_stop_thresh, angles_to_edit, inital_learning_rate
        # Flip:
        #   Num_link_to_render, edit_angles


        if self.do_angles == 'SL':
            lookup = ['lookup']
            s_flip = ['s_flip', 4]
            s_sweep_narrow = ['smartsweep', 10, 4, .1, 'S']
            l_sweep_narrow = ['smartsweep', 10, 4, .1, 'L']
            sl_fine_tune = ['descent',40,4,0.5,.0075,'SL',[.05,.05,None,None,None,None]]

            base_sweep = ['tensorsweep', 8, 4, None, 'S']

            flips = [s_flip]
            sweeps = [l_sweep_narrow,s_sweep_narrow]

            self.stages = [lookup, *flips, *sweeps, *flips, *sweeps, *flips, sl_fine_tune]
            self.stages = [lookup, *flips, *sweeps, *flips]
            self.stages = [lookup, *flips]


        elif self.do_angles == 'SLU':

            # lookup = ['lookup']
            # u_sweep_wide = ['tensorsweep', 50, 6, None, 'U']
            # u_sweep_gen = ['tensorsweep', 50, 6, .3, 'U']
            # u_sweep_narrow = ['smartsweep', 10, 6, .1, 'U']
            # u_stage = ['descent',30,6,0.5,.1,'U',[0.1,0.1,0.4,0.5,0.5,0.5]]
            # s_flip_check_6 = ['s_flip', 6]
            # slu_fine_tune_mandatory = ['descent',40,6,0.5,.0075,'SLU',[None,None,None,None,None,None]]
            # slu_fine_tune_optional = ['descent',35,6,0.5,.0075,'SLU',[None,None,None,None,None,None]]

            # u_sweep_coarse = ['smartsweep', 15, 6, None, 'U']
            # s_sweep = ['smartsweep', 45, 6, 1, 'S']

            # self.stages = [lookup, u_sweep_coarse, s_flip_check_6, s_sweep, s_flip_check_6, u_sweep_narrow, s_flip_check_6, u_stage, s_flip_check_6, slu_fine_tune_mandatory, slu_fine_tune_optional]
            
            # lookup = ['lookup']
            # u_sweep_wide = ['tensorsweep', 50, 5, None, 'U']
            # u_sweep_gen = ['tensorsweep', 50, 5, .3, 'U']
            # u_sweep_narrow = ['smartsweep', 10, 5, .1, 'U']
            # u_stage = ['descent',30,5,0.5,.1,'U',[0.1,0.1,0.4,0.5,0.5,0.5]]
            # s_flip_check_6 = ['s_flip', 5]
            # slu_fine_tune_mandatory = ['descent',40,5,0.5,.0075,'SLU',[None,None,None,None,None,None]]
            # slu_fine_tune_optional = ['descent',35,5,0.5,.0075,'SLU',[None,None,None,None,None,None]]

            # u_sweep_coarse = ['smartsweep', 20, 5, None, 'U']
            # s_sweep = ['smartsweep', 45, 5, 1, 'S']


            # self.stages = [lookup, s_flip_check_6, u_sweep_coarse, s_flip_check_6, s_sweep, s_flip_check_6, u_sweep_narrow, s_flip_check_6, u_stage, s_flip_check_6, slu_fine_tune_mandatory]
            # self.stages = [lookup, s_flip_check_6, u_sweep_coarse, s_flip_check_6, u_sweep_narrow, s_flip_check_6, u_stage, s_flip_check_6, slu_fine_tune_mandatory]


            lookup = ['lookup']

            s_flip_4 = ['s_flip', 4]
            sl_tune = ['descent',10,4,0.5,.1,'SL',[0.05,0.05,0.1,0.5,0.5,0.5]]

            sl_init = [s_flip_4, sl_tune, s_flip_4]

            u_sweep_wide = ['smartsweep', 25, 6, None, 'U']
            s_flip_6 = ['s_flip', 6]
            u_sweep_narrow = ['smartsweep', 10, 5, .1, 'U']
            
            u_stages = [u_sweep_wide, s_flip_4, s_flip_6, u_sweep_narrow]
            
            full_tune = ['descent',40,5,0.5,.0075,'SLU',[None,None,None,None,None,None]]

            self.stages = [lookup, *sl_init, *u_stages, full_tune]




        elif self.do_angles == 'SLUB':

            lookup = ['lookup']
            u_sweep_wide = ['tensorsweep', 50, 6, None, 'U']
            u_sweep_gen = ['tensorsweep', 50, 6, .3, 'U']
            u_sweep_narrow = ['smartsweep', 10, 6, .1, 'U']
            u_stage = ['descent',30,6,0.5,.1,'U',[0.1,0.1,0.4,0.5,0.5,0.5]]
            s_flip_check_6 = ['s_flip', 6]
            slu_fine_tune = ['descent',10,6,0.4,.015,'SLU',[None,None,None,None,None,None]]
            b_sweep_full = ['tensorsweep', 40, 6, None, 'B']
            b_sweep = ['tensorsweep', 5, 6, .1, 'B']
            b_fine_tune = ['descent',5,6,0.4,.015,'B',[None,None,None,.005,.005,None]]
            full_tune = ['descent',10,6,0.4,.015,'SLUB',[None,None,None,None,None,None]]

            self.stages = [lookup, u_sweep_wide, u_sweep_gen, u_sweep_narrow, u_stage, s_flip_check_6, slu_fine_tune,
                b_sweep_full, b_sweep, b_fine_tune, full_tune]

        elif self.do_angles == 'SLURB':

            lookup = ['lookup']
            u_sweep_wide = ['tensorsweep', 50, 6, None, 'U']
            u_sweep_gen = ['tensorsweep', 50, 6, .3, 'U']
            u_sweep_narrow = ['smartsweep', 10, 6, .1, 'U']
            u_stage = ['descent',30,6,0.5,.1,'U',[0.1,0.1,0.4,0.5,0.5,0.5]]
            s_flip_check_6 = ['s_flip', 6]
            slu_fine_tune = ['descent',10,6,0.4,.015,'SLU',[None,None,None,None,None,None]]
            rb_sweep_full = ['tensorsweep', 40, 6, None, 'RB']
            rb_sweep = ['tensorsweep', 5, 6, .1, 'RB']
            rb_fine_tune = ['descent',5,6,0.4,.015,'RB',[None,None,None,.005,.005,None]]
            full_tune = ['descent',10,6,0.4,.015,'SLURB',[None,None,None,None,None,None]]

            self.stages = [lookup, u_sweep_wide, u_sweep_gen, u_sweep_narrow, u_stage, s_flip_check_6, slu_fine_tune,
                rb_sweep_full, rb_sweep, rb_fine_tune, full_tune]






    def run(self, target_color, target_depth, camera_pose = None):
        if camera_pose is not None and np.any(camera_pose != self.camera_pose):
            self.changeCameraPose(camera_pose)


        target_depth = self._downsample(target_depth, self.ds_factor)

        if self.synthetic:
            output, target_depth, lookup_depth = self._loadSynthetic(target_color,target_depth)
        else:
            output, target_depth, lookup_depth = self._segmentLoad(target_color,target_depth)

        if self.preview:
            self.viz.loadTargetColor(target_color)
            self.viz.loadTargetDepth(target_depth)
            self.viz.loadSegmentedLinks(output)

        angle_learning_rate = np.zeros(6)

        history = np.zeros((self.history_length, 6))
        err_history = np.zeros(self.history_length)
        angles = np.array([0]*6, dtype=float)

        self._setStages()
        self.renderer.setJointAngles(angles)

        def preview_if_applicable(color, depth):
            if self.preview:
                self.viz.loadRenderedColor(color)
                self.viz.loadRenderedDepth(depth)
                self.viz.show()

        def render_at_pos(angs):
            self.renderer.setJointAngles(angs)
            return self.renderer.render()

        for stage in self.stages:

            if stage[0] == 'lookup':

                # diff = self._tgt_depth_stack_half - self.lookup_depth
                # diff = tf.abs(diff)
                # lookup_err = tf.reduce_mean(diff, (1,2)) *- tf.math.reduce_std(diff, (1,2))

                diff = self._tgt_depth_stack_full - tf.constant(self.lookup_depth,tf.float32)
                diff = tf.abs(diff)
                lookup_err = (tf.reduce_mean(diff, (1,2)) * tf.math.reduce_std(diff, (1,2))).numpy()

                angles = self.lookup_angles[tf.argmin(lookup_err).numpy()]

            elif stage[0] == 'descent':

                for i in range(6):
                    if stage[6][i] is not None:
                        angle_learning_rate[i] = stage[6][i]

                do_ang = str_to_arr(stage[5])
                self.renderer.setMaxParts(stage[2])

                for i in range(stage[1]):
                    for idx in np.where(do_ang)[0]:

                        if abs(np.mean(history,0)[idx] - angles[idx]) <= angle_learning_rate[idx]:
                            angle_learning_rate[idx] *= stage[3]

                        angle_learning_rate = np.max((angle_learning_rate,self.min_ang_inc),0)

                        def in_limits(ang):
                            return ang >= self.u_reader.joint_limits[idx][0] and ang <= self.u_reader.joint_limits[idx][1]

                        # Under
                        temp = angles.copy()
                        temp[idx] -= angle_learning_rate[idx]
                        if in_limits(temp[idx]):
                            color, depth = render_at_pos(temp)
                            under_err = self._error(stage[2], color, depth)
                            preview_if_applicable(color, depth)

                        else:
                            under_err = np.inf

                        # Over
                        temp[idx] += 2 * angle_learning_rate[idx]
                        if in_limits(temp[idx]):
                            color, depth = render_at_pos(temp)
                            over_err = self._error(stage[2], color, depth)
                            preview_if_applicable(color, depth)
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
                    if ((history.max(0) - history.min(0) <= self.min_ang_inc) + np.isclose((history.max(0) - history.min(0)),self.min_ang_inc)).all():
                        break
                    if (history[:3] == history[0]).all():
                        break

            elif stage[0] == 's_flip':

                self.renderer.setMaxParts(stage[1])

                color, depth = render_at_pos(angles)

                preview_if_applicable(color, depth)
                base_err = self._error(stage[1], color, depth)

                temp = angles.copy()
                temp[0] = -temp[0] + 2*self.camera_pose[5]*np.sign(temp[0])

                limit_thresh = 0.15
                close_to_limits = limit_thresh > abs(self.u_reader.joint_limits[0,0] - temp[0]) or limit_thresh > abs(self.u_reader.joint_limits[0,1] - temp[0])
                _in_limits = temp[0] >= self.u_reader.joint_limits[0,0] and temp[0] <= self.u_reader.joint_limits[0,1]

                # If it's in limits, test it
                if _in_limits:
                    color, depth = render_at_pos(temp)
                    err = self._error(stage[1], color, depth)

                    if err < base_err:
                        angles = temp
                        base_err = err

                        if self.preview:
                            color, depth = render_at_pos(angles)
                            preview_if_applicable(color, depth)
                
                if not _in_limits or close_to_limits:
                    # If out of limits or close to limits, test endpoints
                    for endpoint in self.u_reader.joint_limits[0]:
                        temp[0] = endpoint
                        color, depth = render_at_pos(temp)
                        err = self._error(stage[1], color, depth)

                    if err < base_err:
                        angles = temp
                        base_err = err

                        if self.preview:
                            color, depth = render_at_pos(angles)
                            preview_if_applicable(color, depth)

            elif stage[0] == 'smartsweep':

                do_ang = str_to_arr(stage[4])
                self.renderer.setMaxParts(stage[2])
                div = stage[1]

                color, depth = render_at_pos(angles)
                base_err = self._error(stage[2], color, depth)

                for idx in np.where(do_ang)[0]:
                    temp_low = angles.copy()
                    temp_high = angles.copy()
                    if stage[3] is None:
                        temp_low[idx] = self.u_reader.joint_limits[idx,0]
                        temp_high[idx] = self.u_reader.joint_limits[idx,1]
                    else:
                        temp_low[idx] = max(temp_low[idx]-stage[3], self.u_reader.joint_limits[idx,0])
                        temp_high[idx] = min(temp_high[idx]+stage[3], self.u_reader.joint_limits[idx,1])

                    space = np.linspace(temp_low, temp_high, div)
                    space_err = []
                    for angs in space:
                        color, depth = render_at_pos(angs)
                        space_err.append(self._error(stage[2], color, depth))
                        preview_if_applicable(color, depth)

                    ang_space = space[:,idx]
                    err_pred = interp1d(ang_space, np.array(space_err), kind='cubic')
                    x = np.linspace(temp_low[idx], temp_high[idx], div*5)
                    predicted_errors = err_pred(x)
                    pred_min_ang = x[predicted_errors.argmin()]

                    angs = angles.copy()
                    angs[idx] = pred_min_ang
                    color, depth = render_at_pos(angs)
                    pred_min_err = self._error(stage[2], color, depth)

                    errs = [base_err, min(space_err), pred_min_err]
                    min_type = errs.index(min(errs))

                    if min_type == 1:
                        angles = space[space_err.index(min(space_err))]
                        err_history[1:] = err_history[:-1]
                        err_history[0] = min(space_err)

                    elif min_type == 2:
                        angles = angs
                        err_history[1:] = err_history[:-1]
                        err_history[0] = pred_min_err

                    history[1:] = history[:-1]
                    history[0] = angles

                    if self.preview:
                        color, depth = render_at_pos(angles)
                        preview_if_applicable(color, depth)

            elif stage[0] == 'tensorsweep':

                do_ang = str_to_arr(stage[4])
                self.renderer.setMaxParts(stage[2])
                div = stage[1]

                for idx in np.where(do_ang)[0]:
                    temp_low = angles.copy()
                    temp_high = angles.copy()
                    if stage[3] is None:
                        temp_low[idx] = self.u_reader.joint_limits[idx,0]
                        temp_high[idx] = self.u_reader.joint_limits[idx,1]
                    else:
                        temp_low[idx] = max(temp_low[idx]-stage[3], self.u_reader.joint_limits[idx,0])
                        temp_high[idx] = min(temp_high[idx]+stage[3], self.u_reader.joint_limits[idx,1])

                    space = np.linspace(temp_low, temp_high, div)
                    depths = np.zeros((div, *self.renderer.resolution))
                    for angs, i in zip(space, range(div)):
                        color, depth = render_at_pos(angs)
                        depths[i] = depth
                        preview_if_applicable(color, depth)

                    lookup_depth = tf.pow(tf.constant(depths,tf.float32),0.5)
                    stack = tf.stack([tf.pow(tf.constant(self._tgt_depth, tf.float32),0.5)]*div)

                    diff = tf.abs(stack - lookup_depth)
                    lookup_err = tf.reduce_mean(diff, (1,2)) *- tf.math.reduce_std(diff, (1,2))

                    angles = space[tf.argmin(lookup_err).numpy()]

                    if self.preview:
                        color, depth = render_at_pos(angles)
                        preview_if_applicable(color, depth)

        return angles


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
                    'confidence':data['scores'][idx],
                    'mask':data['masks'][...,idx]
                    }
            else:
                out[self.classes[id]]['mask'] += data['masks'][...,idx]
                out[self.classes[id]]['confidence'] = max(out[self.classes[id]]['confidence'], data['scores'][idx])
        return out

    def _load_target(self, seg_data: dict, tgt_depth: np.ndarray, opt_depth = None) -> None:
        self._masked_targets, self._target_masks = [{} for i in range(2)]
        self._tgt_depth = tgt_depth
        if opt_depth is not None:
            d = opt_depth
        else:
            d = tgt_depth

        # Load cropped lookup info into GPU
        self._tgt_depth_stack_full = tf.stack([tf.constant(applyCrop(d,self.crops[LOOKUP_NUM_RENDERED]), tf.float32)]*len(self.lookup_angles))

        for link in self.link_names:
            if link in seg_data.keys():
                link_mask = seg_data[self.link_names[self.link_names.index(link)]]['mask']
                target_masked = link_mask * tgt_depth
                self._masked_targets[link] = target_masked
                self._target_masks[link] = link_mask

    def _segmentLoad(self, target_color, target_depth):
        r, output = self.seg.segmentImage(self._downsample(target_color, self.ds_factor), process_frame=True)
        segmentation_data = self._reorganize_by_link(r)

        dilate_by = 8
        erode_by = 7

        dilate_by, erode_by = np.ones((dilate_by,dilate_by)), np.ones((erode_by,erode_by))

        # Isolate depth to be only where robot body is
        new = np.zeros((target_depth.shape))
        for k in segmentation_data:
            new += segmentation_data[k]['mask']
        new = cv2.erode(cv2.dilate(new,dilate_by),erode_by)
        target_depth *= new.astype(bool).astype(float)

        # Do same for lookup, but only where appropriate links are
        lookup_depth = target_depth.copy()
        new = np.zeros((target_depth.shape))
        for k in segmentation_data:
            if k in self.u_reader.mesh_names[:LOOKUP_NUM_RENDERED]:
                new += segmentation_data[k]['mask']
        new = cv2.erode(cv2.dilate(new,dilate_by),erode_by)
        lookup_depth *= new.astype(bool).astype(float)

        self._load_target(segmentation_data, target_depth, lookup_depth)

        return output, target_depth, lookup_depth


    def _loadSynthetic(self, target_color, target_depth):
        target_color = self._downsample(target_color, self.ds_factor)

        # Isolate for lookup
        lookup_depth = target_depth.copy()
        new = np.zeros((target_depth.shape))
        for k in self.color_dict:
            if k in self.u_reader.mesh_names[:LOOKUP_NUM_RENDERED]:
                new += target_color[...,0] == self.color_dict[k][0]
        lookup_depth *= new.astype(bool).astype(float)

        self._masked_targets, self._target_masks = [{} for i in range(2)]
        self._tgt_depth = target_depth

        # Load cropped lookup info into GPU
        self._tgt_depth_stack_full = tf.stack([tf.constant(applyCrop(lookup_depth,self.crops[LOOKUP_NUM_RENDERED]), tf.float32)]*len(self.lookup_angles))

        for link in self.link_names:
            link_mask = target_color[...,0] == self.color_dict[link][0]
            if np.sum(link_mask.astype(float)) > 0:
                target_masked = link_mask * target_depth
                self._masked_targets[link] = target_masked
                self._target_masks[link] = link_mask

        return target_color, target_depth, lookup_depth





    def _error(self, num_joints: int, render_color: np.ndarray, render_depth: np.ndarray) -> float:
        color_dict = self.renderer.color_dict
        err = 0

        # Matched Error
        for link in self.link_names[1:num_joints]:
            if link in self._masked_targets.keys():
                target_masked = self._masked_targets[link]
                joint_mask = self._target_masks[link]

                # NOTE: Instead of matching color, this matches blue values,
                #       as each of the default colors has a unique blue value when made.
                render_mask = render_color[...,0] == color_dict[link][0]
                render_masked = render_depth * render_mask

                # Mask
                diff = joint_mask != render_mask
                err += np.mean(diff) * 5

                # Only do if enough depth data present (>5% of required pixels have depth data)
                if np.sum(target_masked != 0) > (.05 * np.sum(joint_mask)):
                    # Depth
                    diff = target_masked - render_masked
                    diff = np.abs(diff) #** .5
                    if diff[diff!=0].size > 0:
                        #err += np.mean(diff[diff!=0]) * np.std(diff[diff!=0])
                        err +=  np.mean(diff[diff!=0]) * 10

        # # Unmatched Error
        # diff = self._tgt_depth - render_depth
        # diff = np.abs(diff) ** 0.5
        # #err += np.mean(diff[diff!=0])
        # err += np.mean(diff[diff!=0]) *- np.std(diff[diff!=0])

        # Unmatched Error
        diff = self._tgt_depth - render_depth
        diff = np.abs(diff)
        #err += np.mean(diff[diff!=0])
        err += np.mean(diff[diff!=0]) * np.std(diff)

        return err







class ProjectionViz():

    def __init__(self, video_path = None, fps = VIDEO_FPS, resolution = (1280, 720)) -> None:
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
        
        out = tgt_d - d
        out[np.where(out == tgt_d)] = 0

        colored = color_array(out)
        colored[np.where(out == tgt_d)] = (55,55,55)

        return colored

    def __del__(self) -> None:
        if self.write_to_file:
            self.writer.release()
