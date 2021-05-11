# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import numpy as np
import h5py

import cv2
from pixellib.instance import custom_segmentation
from scipy.interpolate import interp1d

from robotpose.simulation.render import SkeletonRenderer
from robotpose.urdf import URDFReader
from ..turbo_colormap import color_array

from tqdm import tqdm


DEFAULT_CAMERA_POSE = [.042,-1.425,.399, -.01,1.553,-.057]

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





class ProjectionMatcherLookup():
    def __init__(self,
        default_camera_pose = DEFAULT_CAMERA_POSE,
        ds_factor = 8,
        preview = False,
        do_angle = np.array([True,True,True,False,False,False]),
        min_angle_inc = np.array([.005]*6),
        history_length = 5
        ):

        self.def_cam_pose = default_camera_pose
        self.ds_factor = ds_factor
        self.preview = preview
        if preview:
            self.viz = ProjectionViz('output/projection_viz.avi')
        self.do_angle = do_angle
        self.min_ang_inc = min_angle_inc
        self.history_length = history_length

        self.u_reader = URDFReader()
        self.renderer = SkeletonRenderer('BASE','seg',default_camera_pose,f'1280_720_color_{self.ds_factor}')

        self.classes = ["BG"]
        self.classes.extend(self.u_reader.mesh_names[:6])
        self.link_names = self.classes[1:]

        self.seg = custom_segmentation()
        self.seg.inferConfig(num_classes=6, class_names=self.classes)
        self.seg.load_model("models/segmentation/multi/B.h5")

        with h5py.File('test.h5','r') as f:
            self.lookup_angles = np.copy(f['angles'])
            self.lookup_depth = np.copy(f['depth'])


    def run(self, og_image, target_img, target_depth, camera_pose = None, starting_point = None):
        if camera_pose is None:
            camera_pose = self.def_cam_pose
        self.renderer.setCameraPose(camera_pose)

        if self.preview:
            self.viz.loadSegmentedBody(target_img)
            self.viz.loadTargetColor(og_image)
            self.viz.loadTargetDepth(target_depth)

        target_img = self._downsample(target_img, self.ds_factor)
        target_depth = self._downsample(target_depth, self.ds_factor)
        r, output = self.seg.segmentImage(self._downsample(og_image, self.ds_factor), process_frame=True)
        segmentation_data = self._reorganize_by_link(r)

        if self.preview:
            self.viz.loadSegmentedLinks(output)

        angle_learning_rate = np.zeros(6)

        history = np.zeros((self.history_length, 6))
        err_history = np.zeros(self.history_length)

        # Stages in form:
        # Lookup:
        # Num_link_to_render
        # Sweep/Smartsweep:
        #   Divisions, Num_link_to_render, offset to render, angles_to_edit
        # Descent: 
        #   Iterations, Num_link_to_render, rate reduction, early_stop_thresh, angles_to_edit, inital_learning_rate
        # Flip: 
        #   Num_link_to_render, edit_angles

        if starting_point is None:
            angles = np.array([0]*6, dtype=float)

            lookup = ['lookup', 4]
            u_sweep = ['smartsweep', 25, 6, None, [False,False,True,False,False,False]]
            u_stage = ['descent',30,6,0.5,.1,[False,False,True,False,False,False],[0.1,0.1,0.4,0.5,0.5,0.5]]
            s_flip_check_6 = ['flip', 6, [True,False,False,False,False,False]]
            sl_sweep_check = ['smartsweep', 4, 6, .25, [True,True,False,False,False,False]]
            lu_fine_tune = ['descent',10,6,0.4,.015,[True,True,True,False,False,False],[None,None,None,None,None,None]]
            
            stages = [lookup, u_sweep, u_stage, s_flip_check_6, sl_sweep_check, lu_fine_tune]
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

                        angle_learning_rate = np.max((angle_learning_rate,self.min_ang_inc),0)

                        # Under
                        temp = angles.copy()
                        temp[idx] -= angle_learning_rate[idx]
                        if temp[idx] >= self.u_reader.joint_limits[idx][0] and temp[idx] <= self.u_reader.joint_limits[idx][1]:
                            self.renderer.setJointAngles(temp)
                            color, depth = self.renderer.render()
                            under_err = self._error(stage[2], color, depth)
                            if self.preview:
                                #self._show(color, depth, target_depth)
                                self.viz.loadRenderedColor(color)
                                self.viz.loadRenderedDepth(depth)
                                self.viz.show()
                        else:
                            under_err = np.inf

                        # Over
                        temp = angles.copy()
                        temp[idx] += angle_learning_rate[idx]
                        if temp[idx] >= self.u_reader.joint_limits[idx,0] and temp[idx] <= self.u_reader.joint_limits[idx,1]:
                            self.renderer.setJointAngles(temp)
                            color, depth = self.renderer.render()
                            over_err = self._error(stage[2], color, depth)
                            if self.preview:
                                #self._show(color, depth, target_depth)
                                self.viz.loadRenderedColor(color)
                                self.viz.loadRenderedDepth(depth)
                                self.viz.show()
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

            elif stage[0] == 'flip':

                do_ang = np.array(stage[2])
                self.renderer.setMaxParts(stage[1])

                for idx in np.where(do_ang)[0]:
                    temp = angles.copy()
                    temp[idx] *= -1
                    if temp[idx] >= self.u_reader.joint_limits[idx,0] and temp[idx] <= self.u_reader.joint_limits[idx,1]:
                        self.renderer.setJointAngles(temp)
                        color, depth = self.renderer.render()
                        err = self._error(stage[1], color, depth)

                        if err < err_history[0]:
                            angles[idx] *= -1

                            if self.preview:
                                self.renderer.setJointAngles(angles)
                                color, depth = self.renderer.render()
                                #self._show(color, depth, target_depth)
                                self.viz.loadRenderedColor(color)
                                self.viz.loadRenderedDepth(depth)
                                self.viz.show()

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
                        space_err.append(self._error(stage[2], color, depth))
                        if self.preview:
                            #self._show(color, depth, target_depth)
                            self.viz.loadRenderedColor(color)
                            self.viz.loadRenderedDepth(depth)
                            self.viz.show()

                    angles = space[space_err.index(min(space_err))]

            elif stage[0] == 'smartsweep':

                do_ang = np.array(stage[4])
                self.renderer.setMaxParts(stage[2])
                div = stage[1]

                self.renderer.setJointAngles(angles)
                color, depth = self.renderer.render()
                base_err = self._error(stage[2], color, depth)

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
                        space_err.append(self._error(stage[2], color, depth))
                        if self.preview:
                            #self._show(color, depth, target_depth)
                            self.viz.loadRenderedColor(color)
                            self.viz.loadRenderedDepth(depth)
                            self.viz.show()

                    ang_space = space[:,idx]
                    err_pred = interp1d(ang_space, np.array(space_err), kind='cubic')
                    x = np.linspace(temp_low[idx], temp_high[idx], div*5)
                    predicted_errors = err_pred(x)
                    pred_min_ang = x[predicted_errors.argmin()]

                    angs = angles.copy()
                    angs[idx] = pred_min_ang
                    self.renderer.setJointAngles(angs)
                    color, depth = self.renderer.render()
                    pred_min_err = self._error(stage[2], color, depth)

                    errs = [base_err, min(space_err), pred_min_err]
                    min_type = errs.index(min(errs))
                    
                    if min_type == 1: 
                        angles = space[space_err.index(min(space_err))]
                    elif min_type == 2:
                        angles = angs

                    if self.preview:
                        self.renderer.setJointAngles(angles)
                        color, depth = self.renderer.render()
                        #self._show(color, depth, target_depth)
                        self.viz.loadRenderedColor(color)
                        self.viz.loadRenderedDepth(depth)
                        self.viz.show()

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
        self._tgt_depth_stack = np.stack([tgt_depth]*len(self.lookup_angles))
        for link in self.link_names:
            if link in seg_data.keys():
                link_mask = seg_data[self.link_names[self.link_names.index(link)]]['mask']
                target_masked = link_mask * tgt_depth
                self._masked_targets[link] = target_masked
                self._target_masks[link] = link_mask

    def _error(self, num_joints: int, render_color: np.ndarray, render_depth: np.ndarray) -> float:

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




class ProjectionViz():

    def __init__(self, video_path = None, fps = 45, resolution = (1280, 720)) -> None:
        self.write_to_file = video_path is not None
        self.resolution = resolution
        if video_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.writer = cv2.VideoWriter(video_path, fourcc, 30, resolution)

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

    def loadSegmentedBody(self, segmented_color: np.ndarray) -> None:
        self.seg_body = segmented_color
        self.input_side_up_to_date = False

    def loadSegmentedLinks(self, segmented_color: np.ndarray) -> None:
        self.seg_links = segmented_color
        self.input_side_up_to_date = False

    def loadRenderedColor(self, render_color: np.ndarray) -> None:
        self.rend_color = render_color

    def loadRenderedDepth(self, render_depth: np.ndarray) -> None:
        self.rend_depth = render_depth

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

        #cv2.imshow("Projection Matcher", frame)
        #cv2.waitKey(1)
        if self.write_to_file:
            self.writer.write(self.frame)

    def _seg(self):
        BODY_ALPHA = .2
        body = cv2.resize(self.seg_body, self.resize_to)
        links = cv2.resize(self.seg_links, self.resize_to)
        return cv2.addWeighted(body, BODY_ALPHA, links, (1-BODY_ALPHA), 0)

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

