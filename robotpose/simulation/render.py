# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import numpy as np

import pyrender
import trimesh

from ..projection import makeIntrinsics
from .render_utils import DEFAULT_COLORS, MeshLoader, cameraFromIntrinsics, makePose, posesFromData, setPoses
from ..skeleton import Skeleton
from ..data import Dataset

from .fwd_kinematics_mh5l import FwdKinematic_MH5L_AllJoints as fwdKinematics


class BaseRenderer(Skeleton):
    
    def __init__(
            self,
            skeleton_name,
            mode = 'key',
            camera_pose = None,
            camera_intrin = '1280_720_color',
            suppress_warnings = False
            ):

        super().__init__(skeleton_name)
        intrin = makeIntrinsics(camera_intrin)

        self.mode = mode
        self.suppress_warnings = suppress_warnings

        ml = MeshLoader()
        ml.load()
        name_list = ml.getNames()
        self.meshes = ml.getMeshes()
         
        if camera_pose is not None:
            c_pose = camera_pose
        else:
            c_pose = [.087,-1.425,.4, 0,1.551,-.025]

        self.scene = pyrender.Scene(bg_color=[0.0,0.0,0.0])  # Make scene

        camera = cameraFromIntrinsics(intrin)
        cam_pose = makePose(*c_pose)
        self.camera_node = self.scene.add(camera, pose=cam_pose)

        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
        self.scene.add(dl, parent_node=self.camera_node) # Add light at camera pose

        # Add in joints
        self.joint_nodes = []
        for mesh,name in zip(self.meshes, name_list):
            self.joint_nodes.append(pyrender.Node(name=name,mesh=mesh))
        for node in self.joint_nodes:
            self.scene.add_node(node)
        self._updateKeypoints()
        self.rend = pyrender.OffscreenRenderer(intrin.width, intrin.height)
        self.node_color_map = {}
        self.setMode(mode)
        


    def render(self):
        self.update()
        self._updateKeypoints()
        return self.rend.render(
            self.scene,
            flags=pyrender.constants.RenderFlags.SEG * (self.mode != 'real'),
            seg_node_map=self.node_color_map
            )

    def render_highlight(self,to_highlight, highlight_color):
        self.update()
        self._updateKeypoints()
        for n in to_highlight:
            self._setNodeColor(n, highlight_color)
        return self.rend.render(
            self.scene,
            flags=pyrender.constants.RenderFlags.SEG * (self.mode != 'real'),
            seg_node_map=self.node_color_map
            )



    def setMode(self, mode):
        valid_modes = ['seg','key','seg_full','real']
        assert mode in valid_modes, f"Mode invalid; must be one of: {valid_modes}"
        self.mode = mode
        self._updateMode()


    def setCameraPose(self, pose):
        setPoses(self.scene, [self.camera_node], [makePose(*pose)])

    def getColorDict(self):
        if self.mode == 'seg':
            out = {}
            for node, color in zip(self.node_color_map.keys(), self.node_color_map.values()):
                out[node.name] = color
            return out
        elif self.mode == 'key':
            out = {}
            for node, color in zip(self.node_color_map.keys(), self.node_color_map.values()):
                if node in self.key_nodes:
                    out[node.name] = color
            return out
        elif self.mode == 'seg_full':
            return {'robot': DEFAULT_COLORS[0]}

    
    def _setNodeColor(self, node_name, color):
        try:
            nodes = {node.name:node for node in self.node_color_map.keys()}
            self.node_color_map[nodes[node_name]] = color
        except KeyError:
            pass

    def _updateKeypoints(self, update_mode = True):
        # Remove old
        if hasattr(self, 'key_nodes'):
            if len(self.key_nodes) > 0:
                for node in self.key_nodes:
                    if self.scene.has_node(node):
                        self.scene.remove_node(node)

        # Add in new
        self.key_nodes = []
        marker = trimesh.creation.cylinder(
            self.data['markers']['radius'],
            height=self.data['markers']['height']
            )
        marker = pyrender.Mesh.from_trimesh(marker)

        try:
            for name in self.keypoints:
                parent = self.keypoint_data[name]['parent_link']
                pose = makePose(*self.keypoint_data[name]['pose'])
                n = self.scene.add(marker, name=name, pose=pose, parent_name=parent)
                self.key_nodes.append(n)
        except ValueError as e:
            if str(e) == 'No parent node with name link_name found':
                if not self.suppress_warnings:
                    raise ValueError('No parent node with name link_name found.'+
                        ' It is likely that the template keypoint .json has not been modified')

        if update_mode:
            self._updateMode()


    def _updateMode(self):

        self.node_color_map = {}

        if self.mode == 'seg':
            for joint, idx in zip(self.joint_nodes, range(len(self.joint_nodes))):
                self.node_color_map[joint] = DEFAULT_COLORS[idx]
        elif self.mode == 'key':
            for keypt, idx in zip(self.key_nodes, range(len(self.key_nodes))):
                self.node_color_map[keypt] = DEFAULT_COLORS[idx]
            for joint in self.joint_nodes:
                self.node_color_map[joint] = DEFAULT_COLORS[-1]
        elif self.mode == 'seg_full':
            for joint in self.joint_nodes:
                self.node_color_map[joint] = DEFAULT_COLORS[0]





class SkeletonRenderer(BaseRenderer):
    
    def __init__(
            self,
            skeleton_name,
            mode = 'key',
            camera_pose = None,
            camera_intrin = '1280_720_color',
            suppress_warnings = False
            ):

        super().__init__(skeleton_name, mode, camera_pose, camera_intrin, suppress_warnings)

    def setJointAngles(self, angles):
        setPoses(self.scene, self.joint_nodes,posesFromData(np.array([angles]), np.array([fwdKinematics(angles)]))[0])






class DatasetRenderer(BaseRenderer):
    
    def __init__(
            self,
            dataset,
            skeleton,
            ds_type = 'full',
            mode = 'seg',
            camera_pose = None,
            camera_intrin = '1280_720_color',
            robot_name="mh5"
            ):

        super().__init__(skeleton, mode, camera_pose, camera_intrin)
        self.ds = Dataset(dataset, skeleton, ds_type = ds_type)


    def setObjectPoses(self, poses):
        setPoses(self.scene, self.joint_nodes, poses)


    def setPosesFromDS(self, idx):
        if not hasattr(self,'ds_poses'):
            self.ds_poses = posesFromData(self.ds.angles, self.ds.positions)
        self.setObjectPoses(self.ds_poses[idx])
        setPoses(self.scene, [self.camera_node], [makePose(*self.ds.camera_pose[idx])])

