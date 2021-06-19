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
import pyrender
import PySimpleGUI as sg

from ..projection import Intrinsics
from .render_utils import DEFAULT_COLORS, MeshLoader, makePose, setPoses
from ..data import Dataset

from .kinematics import ForwardKinematics


class Renderer():
    
    def __init__(
            self,
            mode = 'seg',
            camera_pose = None,
            camera_intrin = '1280_720_color',
            suppress_warnings = False,
            intrinsic_ds_factor = None
            ):

        self.kine = ForwardKinematics()

        self.intrinsics = Intrinsics(camera_intrin)
        if intrinsic_ds_factor is not None:
            self.intrinsics.downscale(intrinsic_ds_factor)

        self.mode = mode
        self.suppress_warnings = suppress_warnings
        self.limit_parts = False
         
        if camera_pose is not None:
            c_pose = camera_pose
        else:
            c_pose = [.087,-1.425,.73, 0,1.551,-.025]

        self.scene = pyrender.Scene(bg_color=[0.0,0.0,0.0])  # Make scene

        camera = self.intrinsics.pyrender_camera
        cam_pose = makePose(*c_pose)
        self.camera_node = self.scene.add(camera, pose=cam_pose)

        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
        self.scene.add(dl, parent_node=self.camera_node) # Add light at camera pose
        self.rend = pyrender.OffscreenRenderer(self.intrinsics.width, self.intrinsics.height)

        self.loadMeshes()        


    def loadMeshes(self):
        ml = MeshLoader()
        self.meshes, name_list = ml.meshes_and_names

        if hasattr(self, 'joint_nodes'):
            if len(self.joint_nodes) > 0:
                for node in self.joint_nodes:
                    self.scene.remove_node(node)

        # Add in joints
        self.joint_nodes = []
        for mesh,name in zip(self.meshes, name_list):
            self.joint_nodes.append(pyrender.Node(name=name,mesh=mesh))
        for node in self.joint_nodes:
            self.scene.add_node(node)
        self.node_color_map = {}
        self.setMode(self.mode)


    def setJointAngles(self, angles):
        setPoses(self.scene, self.joint_nodes,self.kine.calc(angles))

    def render(self):
        return self.rend.render(
            self.scene,
            flags=pyrender.constants.RenderFlags.SEG * (self.mode != 'real'),
            seg_node_map=self.node_color_map
            )

    def setMode(self, mode):
        valid_modes = ['seg','seg_full','real']
        assert mode in valid_modes, f"Mode invalid; must be one of: {valid_modes}"
        self.mode = mode
        self._updateMode()

    def setCameraPose(self, pose_in):
        pose = np.copy(pose_in)
        pose[4] += np.pi/2
        setPoses(self.scene, [self.camera_node], [makePose(*pose)])

    def _setNodeColor(self, node_name, color):
        try:
            nodes = {node.name:node for node in self.node_color_map.keys()}
            self.node_color_map[nodes[node_name]] = color
        except KeyError:
            pass

    def setMaxParts(self, number_of_parts):
        if number_of_parts is not None:
            self.limit_parts = True
            self.limit_number = number_of_parts
        else:
            self.limit_parts = False
        self._updateMode()

    def _updateMode(self):

        self.node_color_map = {}

        if self.mode == 'seg':
            if self.limit_parts:
                for joint, idx in zip(self.joint_nodes[:self.limit_number], range(self.limit_number)):
                    self.node_color_map[joint] = DEFAULT_COLORS[idx]
            else:
                for joint, idx in zip(self.joint_nodes, range(len(self.joint_nodes))):
                    self.node_color_map[joint] = DEFAULT_COLORS[idx]
        elif self.mode == 'seg_full':
            for joint in self.joint_nodes:
                self.node_color_map[joint] = DEFAULT_COLORS[0]

    @property
    def resolution(self):
        return (self.rend.viewport_height, self.rend.viewport_width)

    @property
    def camera_pose(self):
        return self.scene.get_pose(self.camera_node)

    @property
    def color_dict(self):
        if self.mode == 'seg':
            out = {}
            for node, color in zip(self.node_color_map.keys(), self.node_color_map.values()):
                out[node.name] = color
            return out
        elif self.mode == 'seg_full':
            return {'robot': DEFAULT_COLORS[0]}



class DatasetRenderer(Renderer):
    
    def __init__(
            self,
            dataset,
            ds_type = 'full',
            mode = 'seg',
            camera_pose = None,
            robot_name="mh5"
            ):

        self.ds = Dataset(dataset, ds_type = ds_type)
        super().__init__(mode, camera_pose, self.ds.attrs['color_intrinsics'])
        

    def setObjectPoses(self, poses):
        setPoses(self.scene, self.joint_nodes, poses)

    def setPosesFromDS(self, idx):
        self.setObjectPoses(self.kine.calc(self.ds.angles[idx]))
        self.setCameraPose(self.ds.camera_pose[idx])



class Aligner():
    """
    Used to manually find the position of camera relative to robot.

    W/S - Move forward/backward
    A/D - Move left/right
    Z/X - Move down/up
    Q/E - Roll
    R/F - Tilt down/up
    G/H - Pan left/right
    +/- - Increase/Decrease Step size
    """

    def __init__(self, dataset, start_idx = None, end_idx = None):
        # Load dataset
        self.ds = Dataset(dataset, permissions='a')

        self.renderer = DatasetRenderer(dataset, mode='seg_full')

        if start_idx is not None:
            self.start_idx = start_idx
        else:
            self.start_idx = 0

        self.idx = self.start_idx

        if end_idx is not None:
            self.end_idx = end_idx
        else:
            self.end_idx = self.ds.length - 1

        self.inc = int((self.end_idx - self.start_idx + 1)/20)
        if self.inc > 10:
            self.inc = 10
        if self.inc < 1:
            self.inc = 1

        self.c_pose = self.ds.camera_pose[self.start_idx]

        # Movement steps
        self.xyz_steps = [.001,.005,.01,.05,.1,.25,.5]
        self.ang_steps = [.0005,.001,.005,.01,.025,.05,.1]
        self.step_loc = len(self.xyz_steps) - 4

        self._findSections()
        self.section_idx = 0

        print("Copying Image Array...")
        self.real_arr = np.copy(self.ds.og_img)
        self.zoom = 1

        self.gui = AlignerGUI()


    def run(self):
        ret = True
        move = True

        while ret:
            event, values = self.gui.update(self.section_starts, self.section_idx)
            if event == 'quit':
                print("Quit by user.")
                ret = False
                continue
            elif event == 'new_section':
                self._newSection(values)
            elif event == 'goto':
                if 0 <= self.idx and self.idx < self.ds.length:
                    self.idx = values
                    move = True
            elif event == 'pose_entry':
                if np.any(values != self.c_pose):
                    self.c_pose = values
                    self.saveCameraPose()
            elif event == 'zoom':
                self.zoom = values

            self._getSection()
            self.readCameraPose()

            if move:
                real = self.real_arr[self.idx]
                self.renderer.setPosesFromDS(self.idx)
                render, depth = self.renderer.render()
                image = self.combineImages(real, render)
                image = self.applyZoom(image)
                move = False
            image = self.addOverlay(image)
            cv2.imshow("Aligner", image)

            inp = cv2.waitKey(1)
            ret, move = self.moveCamera(inp)

        self.gui.close()
        cv2.destroyAllWindows()


    def moveCamera(self,inp):
        xyz_step = self.xyz_steps[self.step_loc]
        ang_step = self.ang_steps[self.step_loc]

        if inp == ord('0'):
            return False, False
        elif inp == ord('='):
            self.step_loc += 1
            if self.step_loc >= len(self.xyz_steps):
                self.step_loc = len(self.xyz_steps) - 1
            return True, False
        elif inp == ord('-'):
            self.step_loc -= 1
            if self.step_loc < 0:
                self.step_loc = 0
            return True, False
        elif inp == ord('k'):
            self.increment(-self.inc)
            return True, True
        elif inp == ord('l'):
            self.increment(self.inc)
            return True, True

        if inp == ord('d'):
            self.c_pose[0] -= xyz_step
        elif inp == ord('a'):
            self.c_pose[0] += xyz_step
        elif inp == ord('w'):
            self.c_pose[1] -= xyz_step
        elif inp == ord('s'):
            self.c_pose[1] += xyz_step
        elif inp == ord('z'):
            self.c_pose[2] += xyz_step
        elif inp == ord('x'):
            self.c_pose[2] -= xyz_step
        elif inp == ord('q'):
            self.c_pose[3] -= ang_step
        elif inp == ord('e'):
            self.c_pose[3] += ang_step
        elif inp == ord('r'):
            self.c_pose[4] -= ang_step
        elif inp == ord('f'):
            self.c_pose[4] += ang_step
        elif inp == ord('g'):
            self.c_pose[5] += ang_step
        elif inp == ord('h'):
            self.c_pose[5] -= ang_step

        self.saveCameraPose()
        return True, True


    def applyZoom(self, image):
        dim = list(image.shape[:2])
        dim.reverse()
        dim = [x* self.zoom for x in dim]
        dim = [int(x) for x in dim]
        image = cv2.resize(image, tuple(dim))
        return image

    def addOverlay(self, image):
        pose_str = "["
        for num in self.c_pose:
            pose_str += f"{num:.3f}, "
        pose_str +="]"
        image = cv2.putText(image, pose_str,(10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        image = cv2.putText(image, str(self.xyz_steps[self.step_loc]),(10,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        image = cv2.putText(image, str(self.ang_steps[self.step_loc]),(10,150), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        image = cv2.putText(image, str(self.idx),(10,200), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        return image


    def combineImages(self,image_a, image_b, weight = .5):
        return np.array(image_a * weight + image_b *(1-weight), dtype=np.uint8)


    def increment(self, step):
        if (self.idx + step >= 0) and (self.idx + step < self.ds.length):
            self.idx += step


    def saveCameraPose(self):
        for idx in range(self.start_idx, self.end_idx + 1):
            self.ds.camera_pose[idx,:] = self.c_pose

    def readCameraPose(self):
        self.c_pose = self.ds.camera_pose[self.idx,:]

    def _findSections(self):
        self.section_starts = []
        p = [0,0,0,0,0,0]
        for idx in range(self.ds.length):
            if not np.array_equal(self.ds.camera_pose[idx], p):
                self.section_starts.append(idx)
                p = self.ds.camera_pose[idx,:]
        self.section_starts.append(self.ds.length)
        return self.section_starts

    def _newSection(self, idx):
        self.section_starts.append(idx)
        self.section_starts.sort()

    def _getSection(self):
        section_start = max([x for x in self.section_starts if x <= self.idx])
        self.section_idx = self.section_starts.index(section_start)
        self.start_idx = section_start
        self.end_idx = self.section_starts[self.section_idx + 1] - 1



class AlignerGUI():

    def __init__(self):
        control_str = "W/S - Move forward/backward\n"+\
            "A/D - Move left/right\nZ/X - Move up/down\nQ/E - Roll\n"+\
            "R/F - Tilt down/up\nG/H - Pan left/right\n+/- - Increase/Decrease Step size\nK/L - Last/Next image"
        self.layout = [[sg.Text("Currently Editing:"), sg.Text(size=(40,1), key='editing')],
                        [sg.Input(size=(5,1),key='num_input'),sg.Button('Go To',key='num_goto'), sg.Button('New Section',key='new_section')],
                        [sg.Text("Manual Pose:"), sg.Input(size=(20,1),key='pose_entry')],
                        [sg.Text("Zoom:"),sg.Slider((.5,4),1,key='zoom',orientation='h', resolution=0.1)],
                        [sg.Text("",key='warn',text_color="red", size=(22,1))],
                        [sg.Table([[["Sections:"]],[[1,1]]], key='sections'),sg.Text(control_str)],
                        [sg.Button('Quit',key='quit')]]

        self.window = sg.Window('Aligner Controls', self.layout, return_keyboard_events = True, use_default_focus=False)
        self.past_zoom = 1

    def update(self, section_starts, section_idx):
        event, values = self.window.read(timeout=1, timeout_key='tm')
        section_table = []
        for idx in range(len(section_starts)-1):
            section_table.append([[f"{section_starts[idx]} - {section_starts[idx+1]-1}"]])

        self.window['editing'].update(f"{section_starts[section_idx]} - {section_starts[section_idx+1]-1}")
        self.window['sections'].update(section_table)

        if values['zoom'] != self.past_zoom:
            self.past_zoom = values['zoom']
            return ['zoom',values['zoom']]

        try:
            if event == 'new_section':
                return ['new_section',int(values['num_input'])]
            elif event == 'num_goto':
                return ['goto',int(values['num_input'])]
            if len(values['num_input']) > 0:
                if int(values['num_input']) is not None:
                    self.window['warn'].update("")
        except ValueError:
            self.window['warn'].update("Please input a number.")

        if event == 'quit':
            self.close()
            return ['quit',None]

        try:
            entry = values['pose_entry'].replace('[','').replace(']','').replace(',',' ')
            entry = np.fromstring(entry,np.float,sep=' ')
            if entry.shape == (6,):
                return ['pose_entry',entry]

        except ValueError:
            pass

        return [None,None]

    def close(self):
        self.window.close()