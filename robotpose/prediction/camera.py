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
from scipy.interpolate import interp1d

from ..simulation.lookup import RobotLookupManager
from ..simulation.render import Renderer
from ..turbo_colormap import color_array
from ..urdf import URDFReader
from ..utils import str_to_arr, get_gpu_memory
from tqdm import tqdm

import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()

# DEFAULT_CAMERA_POSE = [.042,-1.425,.75, -.01,1.553,-.057]
DEFAULT_CAMERA_POSE = [0, -1.5, .75, 0, 0, 0]

class ModellessCameraPredictor():
    def __init__(self,
        base_pose = DEFAULT_CAMERA_POSE,
        ds_factor: int = 8,
        preview: bool = False,
        save_to: str = None,
        min_angle_inc = np.array([0.001,0.001,0.001,0.002,0.002,0.002]),
        history_length = 5
        ):

        self.base_pose = np.array(base_pose)

        self.ds_factor = ds_factor
        self.preview = preview
        if preview:
            self.viz = ModellessProjectionViz(save_to)
        self.min_ang_inc = min_angle_inc
        self.history_length = history_length

        self.intrinsics = f'1280_720_color_{self.ds_factor}'

        self.u_reader = URDFReader()
        self.renderer = Renderer('seg',None,self.intrinsics)

        self.classes = ["BG"]
        self.classes.extend(self.u_reader.mesh_names[:6])
        self.link_names = self.classes[1:]

        self.renderer.setMaxParts(None)


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
        # zp_sweep:
        #   Divisions, range (one-sided)


        coarse_descent = ['descent', 50, 0.5, .01, [True]*6, [0.1,0.1,0.1,0.05,0.05,0.05]]
        wide_tensorsweep_xyz = ['tensorsweep', 20, .2, [True,True,True,False,False,False]]
        wide_tensorsweep_rpy = ['tensorsweep', 20, .1, [False,False,False,True,True,True]]
        fine_descent = ['descent', 50, 0.5, .001, [True]*6, [0.01,0.01,0.01,0.01,0.01,0.01]]
        # tilt_fix = ['descent', 100, 0.75, .00001, [False,False,True,False,True,False], [None,None,0.1,None,0.05,None]]
        # z_sweep = ['smartsweep', 10, .2, [False,False,True,False,False,False]]
        # p_sweep = ['smartsweep', 10, .1, [False,False,False,False,True,False]]
        # u_sweep_narrow = ['smartsweep', 10, .1, [False,False,True,False,False,False]]

        zp_sweep = ['zp_sweep', 20, 0.1]
        p_fix = ['smartsweep', 20, .03, [False,False,False,False,True,False]]
        xyya_narrow = ['smartsweep', 20, .15, [True,True,False,False,False,True]]*10
        quick_descent = ['descent', 15, 0.5, .001, [True]*6, [0]*6]

        xya_sweep = ['xya_sweep', 20, 0.1]
        ya_fix = p_fix = ['smartsweep', 20, .03, [False,False,False,False,False,True]]

        combo = [zp_sweep,p_fix,xyya_narrow]*2
        combo_2 = [xya_sweep,ya_fix]*2

        coarse_descent = ['descent', 50, 0.5, .01, [True]*6, [0.1,0.1,0.1,0.05,0.05,0.05]]

        coarse_replacement = []
        [coarse_replacement.extend([['tensorsweep', 20, x, [True,True,True,False,False,False]],['tensorsweep', 20, x/2, [False,False,False,True,True,True]]]) for x in np.logspace(1,.05,5)/30]

        # rb_fine_tune = ['descent', 5, 0.4, .015, [False,False,False,True,True,False], [None,None,None,.005,.005,None]]
        # full_tune = ['descent', 10, 0.4, .015, [True,True,True,True,True,False], [None,None,None,None,None,None]]

        #self.stages = [coarse_descent, wide_tensorsweep_xyz, wide_tensorsweep_rpy, fine_descent, zp_sweep, p_fix, xyya_narrow, quick_descent]
        #self.stages = [coarse_descent, wide_tensorsweep_xyz, wide_tensorsweep_rpy, fine_descent, *combo, quick_descent]
        self.stages = [*(coarse_replacement), wide_tensorsweep_xyz, wide_tensorsweep_rpy, fine_descent, *combo, quick_descent,*combo_2,quick_descent]
        #self.stages = [['spiral']]



    def do_renders_at_pose(self, pose):
        self.renderer.setCameraPose(pose)
        color_out = np.zeros((self.number_of_poses,*self.renderer.resolution,3))
        depth_out = np.zeros((self.number_of_poses,*self.renderer.resolution))
        for idx in range(self.number_of_poses):
            self.renderer.setJointAngles(self.robot_poses[idx])
            color_out[idx], depth_out[idx] = self.renderer.render()

        return color_out, depth_out

    def run(self, og_images, target_depths, robot_poses, starting_camera_pose = None):
        if len(og_images.shape) == 3:
            og_images = np.array([og_images])
            target_depths = np.array([target_depths])
            robot_poses = np.array([robot_poses])
        self.robot_poses = np.array(robot_poses)
        assert og_images.shape[0] == target_depths.shape[0] == self.robot_poses.shape[0]

        # # Filter Depths
        # target_depths[target_depths > 3] = 0
        # target_depths[target_depths < 0.5] = 0

        self.number_of_poses = og_images.shape[0]

        if self.preview:
            self.viz.loadTargetColor(og_images[0])
            self.viz.loadTargetDepth(target_depths[0])

        self._tgt_depths = self._batch_downsample(target_depths, self.ds_factor)
        og_images = self._batch_downsample(og_images, self.ds_factor)

        learning_rates = np.zeros(6)

        history = np.zeros((self.history_length, 6))
        err_history = np.zeros(self.history_length)

        if starting_camera_pose is None:
            pose = np.copy(self.base_pose)
        else:
            pose = np.array(starting_camera_pose)

        self._setStages()


        def preview_if_applicable(color, depth):
            if len(color.shape) == 4:
                color = color[0]
                depth = depth[0]
            if self.preview:
                self.viz.loadRenderedColor(color)
                self.viz.loadRenderedDepth(depth)
                self.viz.show()



        for stage in self.stages:
            print(pose, f"starting {stage[0]}")

            if stage[0] == 'spiral':

                sp = SpiralRenderer(self.renderer, self._error, self.do_renders_at_pose)
                if self.preview:
                    sp.loadVisualizer(self.viz)

                pose = sp.run()
                print(pose)

                # diff = self._tgt_depth_stack_half - self.lookup_depth
                # diff = tf.abs(diff)
                # lookup_err = tf.reduce_mean(diff, (1,2)) *- tf.math.reduce_std(diff, (1,2))

                # pose = self.lookup_angles[tf.argmin(lookup_err).numpy()]
            elif stage[0] == 'descent':

                for i in range(6):
                    if stage[5][i] is not None:
                        learning_rates[i] = stage[5][i]

                do_param = np.array(stage[4])

                for i in range(stage[1]):
                    for idx in np.where(do_param)[0]:

                        if abs(np.mean(history,0)[idx] - pose[idx]) <= learning_rates[idx]:
                            learning_rates[idx] *= stage[2]
                            #print(f"{['x ','y ','z ','ro','pi','ya'][idx]} rate reduction to {learning_rates[idx]}")

                        learning_rates = np.max((learning_rates,self.min_ang_inc),0)

                        # Under
                        temp = pose.copy()
                        temp[idx] -= learning_rates[idx]
                        color, depth = self.do_renders_at_pose(temp)
                        under_err = self._error(depth)
                        preview_if_applicable(color, depth)

                        # Over
                        temp[idx] += 2 * learning_rates[idx]
                        color, depth = self.do_renders_at_pose(temp)
                        over_err = self._error(depth)
                        preview_if_applicable(color, depth)

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

            elif stage[0] == 'smartsweep':

                do_param = np.array(stage[3])
                div = stage[1]

                color, depth = self.do_renders_at_pose(pose)
                base_err = self._error(depth)

                for idx in np.where(do_param)[0]:
                    temp_low = pose.copy()
                    temp_high = pose.copy()

                    temp_low[idx] = temp_low[idx]-stage[2]
                    temp_high[idx] = temp_low[idx]+stage[2]

                    space = np.linspace(temp_low, temp_high, div)
                    space_err = []
                    for pose_val in space:
                        color, depth = self.do_renders_at_pose(pose_val)
                        space_err.append(self._error(depth))
                        preview_if_applicable(color, depth)

                    pose_space = space[:,idx]
                    err_pred = interp1d(pose_space, np.array(space_err), kind='cubic')
                    x = np.linspace(temp_low[idx], temp_high[idx], div*5)
                    predicted_errors = err_pred(x)
                    pred_min_ang = x[predicted_errors.argmin()]

                    temp_pose = pose.copy()
                    temp_pose[idx] = pred_min_ang
                    color, depth = self.do_renders_at_pose(temp_pose)
                    pred_min_err = self._error(depth)

                    errs = [base_err, min(space_err), pred_min_err]
                    min_type = errs.index(min(errs))

                    if min_type == 1:
                        pose = space[space_err.index(min(space_err))]
                        err_history[1:] = err_history[:-1]
                        err_history[0] = min(space_err)

                    elif min_type == 2:
                        pose = temp_pose
                        err_history[1:] = err_history[:-1]
                        err_history[0] = pred_min_err

                    history[1:] = history[:-1]
                    history[0] = pose

                    if self.preview:
                        color, depth = self.do_renders_at_pose(pose)
                        preview_if_applicable(color, depth)

            elif stage[0] == 'tensorsweep':

                do_param = np.array(stage[3])
                div = stage[1]

                for idx in np.where(do_param)[0]:
                    temp_low = pose.copy()
                    temp_high = pose.copy()

                    temp_low[idx] -= stage[2]
                    temp_high[idx] += stage[2]

                    space = np.linspace(temp_low, temp_high, div)
                    depths = np.zeros((div, self.number_of_poses, *self.renderer.resolution))
                    for temp_pose, i in zip(space, range(div)):
                        color, depth = self.do_renders_at_pose(temp_pose)
                        depths[i] = depth
                        preview_if_applicable(color, depth)


                    errs = self._error(depths)
                    pose = space[errs.argmin()]

                    if self.preview:
                        color, depth = self.do_renders_at_pose(pose)
                        preview_if_applicable(color, depth)

            elif stage[0] == 'zp_sweep':
                #stage[1] is div
                #stage[2] is range
                temp_low = pose.copy()
                temp_high = pose.copy()

                div = stage[1]

                temp_low = pose.copy()
                temp_high = pose.copy()

                temp_pose = pose.copy()

                temp_low[2] = temp_pose[2]-stage[2]
                temp_high[2] = temp_pose[2]+stage[2]

                space = np.linspace(temp_low, temp_high, div)
                space[:,4] = np.arctan(np.tan(temp_pose[4]) - ((space[:,2] - temp_pose[2]) / np.sqrt(temp_pose[0] ** 2 + temp_pose[1] ** 2)))

                # Using Tensorflow
                depths = np.zeros((div, self.number_of_poses, *self.renderer.resolution))
                for temp_pose, i in zip(space, range(div)):
                    color, depth = self.do_renders_at_pose(temp_pose)
                    depths[i] = depth
                    preview_if_applicable(color, depth)

                errs = self._error(depths)
                pose = space[errs.argmin()]

            elif stage[0] == 'xya_sweep':
                #stage[1] is div
                #stage[2] is range
                temp_low = pose.copy()
                temp_high = pose.copy()

                div = stage[1]

                temp_pose = pose.copy()

                temp_low[0] = temp_pose[0]-stage[2]
                temp_high[0] = temp_pose[0]+stage[2]

                space = np.linspace(temp_low, temp_high, div)
                space[:,5] = -np.arctan(((space[:,0] - pose[0] )/ pose[0]) * np.tan(pose[5]))

                # Using Tensorflow
                depths = np.zeros((div, self.number_of_poses, *self.renderer.resolution))
                for temp_pose, i in zip(space, range(div)):
                    color, depth = self.do_renders_at_pose(temp_pose)
                    depths[i] = depth
                    preview_if_applicable(color, depth)

                errs = self._error(depths)
                pose = space[errs.argmin()]

        return pose


    def _batch_downsample(self, base: np.ndarray, factor: int) -> np.ndarray:
        dims = [x//factor for x in base.shape[1:3]]

        if len(base.shape) == 4:
            out = np.zeros((base.shape[0],*dims,3))
        else:
            out = np.zeros((base.shape[0],*dims))
        dims.reverse()
        for idx in range(base.shape[0]):
            out[idx] = cv2.resize(base[idx], tuple(dims))
        return out


    def _error(self, render_depth_frames: np.ndarray) -> float:

        power = 0.5

        if len(render_depth_frames.shape) == 4:

            # div, poses, height, width
            div = render_depth_frames.shape[0]

            rendered = tf.pow(tf.constant(render_depth_frames,tf.float32),power)
            actual = tf.stack([tf.pow(tf.constant(self._tgt_depths, tf.float32),power)]*div)

            # actual = actual * tf.cast((rendered != 0),float)
            # rendered = rendered * tf.cast((actual != 0),float)

            diff = tf.abs(actual - rendered)
            err = tf.reduce_mean(diff, (2,3)) *- tf.math.reduce_std(diff, (2,3))
            #err = tf.pow(err,2) * tf.abs(err) / err
            err = tf.pow(1.1,err)
            err = tf.reduce_mean(err,1).numpy()

        else:

            #poses, height, width

            rendered = tf.pow(tf.constant(render_depth_frames,tf.float32),power)
            actual = tf.pow(tf.constant(self._tgt_depths, tf.float32),power)

            # actual = actual * tf.cast((rendered != 0),float)
            # rendered = rendered * tf.cast((actual != 0),float)

            diff = tf.abs(actual - rendered)
            err = tf.reduce_mean(diff, (1,2)) *- tf.math.reduce_std(diff, (1,2))
            #err = tf.pow(err,2) * tf.abs(err) / err
            err = tf.pow(1.1,err)
            err = tf.reduce_mean(err).numpy()


        return err

    def error_at(self, pose):
        color, depth = self.do_renders_at_pose(pose)
        return self._error(depth)


class SpiralRenderer():
    def __init__(self, renderer, error_func, render_func, batch = 10000, r_limits = [1,3], shells = 25, per_round = 75, z_limits = [0,1], turns = 10) -> None:
        self.renderer = renderer
        self.error = error_func
        self.batch = batch
        self.r_min = min(r_limits)
        self.r_max = max(r_limits)
        self.shells = shells
        self.per_round = per_round
        self.z_min = min(z_limits)
        self.z_max = max(z_limits)
        self.turns = turns
        self.visualization = False
        self.render_func = render_func

    def loadVisualizer(self, viz):
        self.viz = viz
        self.visualization = True

    def run(self):

        # Make the base spiral
        num_per_spiral = self.turns * self.per_round
        base_spiral = np.zeros((num_per_spiral,6))
        angles_partial = np.linspace(0, 2*np.pi, self.per_round)
        angles_full = np.tile(angles_partial, self.turns)
        base_spiral[:,5] = 2*np.pi - angles_full
        base_spiral[:,0] = -np.sin(angles_full)
        base_spiral[:,1] = -np.cos(angles_full)
        base_spiral[:,2] = np.linspace(self.z_min, self.z_max, num_per_spiral)

        # Convert to shells
        full_space = np.tile(base_spiral,(self.shells,1))
        r_partial = np.linspace(self.r_min, self.r_max, self.shells)
        r_full = np.repeat(r_partial, num_per_spiral)
        full_space[:,0] *= r_full
        full_space[:,1] *= r_full

        end_pts = np.arange(0, full_space.shape[0], self.batch)
        if end_pts[-1] != full_space.shape[0]:
            end_pts = np.append(end_pts, full_space.shape[0])

        errors = np.zeros(full_space.shape[0])

        # print(full_space[-1])
        # full_space[:,:] = [3424,234423,3424]

        for batch in range(end_pts.shape[0]-1):
            start_idx = end_pts[batch]
            end_idx = end_pts[batch + 1]

            for pose,idx in tqdm(zip(full_space[start_idx:end_idx],range(start_idx,end_idx)), total = end_idx - start_idx):
                self.render_func(pose)
                color, depths = self.renderer.render()
                errors[idx] = self.error(depths)
                if self.visualization:
                    self.viz.loadRenderedColor(color)
                    self.viz.loadRenderedDepth(depths)
                    self.viz.show()

        plt.plot(errors)
        plt.show()

        return full_space[errors.argmin()]




class ModellessProjectionViz():

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

    def loadRenderedColor(self, render_color: np.ndarray) -> None: self.rend_color = render_color

    def loadRenderedDepth(self, render_depth: np.ndarray) -> None: self.rend_depth = render_depth

    def _genInput(self):
        self.frame[:self.res[0]//2, :self.res[1]//2] = self._orig()

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
