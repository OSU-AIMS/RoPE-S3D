# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

from typing import List, Tuple, Union

import cv2
import numpy as np
import pyrender
import PySimpleGUI as sg

from ..constants import DEFAULT_RENDER_COLORS
from ..data import Dataset
from ..projection import Intrinsics
from .kinematics import ForwardKinematics
from .render_utils import MeshLoader, makePose, setPoses


class Renderer():
    
    def __init__(
            self,
            mode: str = 'seg',
            camera_pose: np.ndarray = None,
            camera_intrin: Union[str, Intrinsics] = '1280_720_color',
            suppress_warnings: bool = False,
            intrinsic_ds_factor: int = None
            ):

        self.mode, self.suppress_warnings = mode ,suppress_warnings
        self.kine = ForwardKinematics()

        # Create and downscale (if applicable)
        self.intrinsics = Intrinsics(camera_intrin)
        if intrinsic_ds_factor is not None:
            self.intrinsics.downscale(intrinsic_ds_factor)

        self.limit_parts = False
         
        if camera_pose is not None:
            c_pose = camera_pose
        else:
            c_pose = [0.04, -1.425, 0.75, 0, -0.02, -0.05]

        # Create scene and camera
        self.scene = pyrender.Scene(bg_color=[0.0,0.0,0.0])  # Make scene
        camera = self.intrinsics.pyrender_camera
        self.camera_node = self.scene.add(camera)
        self.setCameraPose(c_pose)

        # Set lighting
        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=10.0)
        self.scene.add(dl, parent_node=self.camera_node) # Add light at camera pose
        self.rend = pyrender.OffscreenRenderer(self.intrinsics.width, self.intrinsics.height)

        self.loadMeshes()        


    def loadMeshes(self):
        """Load robot meshes as nodes"""
        ml = MeshLoader()
        self.meshes, name_list = ml.meshes_and_names

        # Clear if already existant
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
        
        # Set node color map
        self.node_color_map = {}
        self.setMode(self.mode)


    def setJointAngles(self, angles: List[float]):
        """Set the position of the robot's links based on the joint angles"""
        setPoses(self.scene, self.joint_nodes,self.kine.calc(angles))

    def render(self):
        """Render the scene using Pyrender"""
        return self.rend.render(
            self.scene,
            flags=pyrender.constants.RenderFlags.SEG * (self.mode != 'real'),
            seg_node_map=self.node_color_map
            )

    def setMode(self, mode: str):
        """Change/Set the rendering mode"""
        valid_modes = ['seg','seg_full','real']
        assert mode in valid_modes, f"Mode invalid; must be one of: {valid_modes}"
        self.mode = mode
        self._updateMode()

    def setCameraPose(self, pose_in: np.ndarray):
        """Set the camera pose, using a convention for camera pitch"""
        pose = np.copy(pose_in)
        pose[4] += np.pi/2
        setPoses(self.scene, [self.camera_node], [makePose(*pose)])

    def _setNodeColor(self, node_name: str, color: List[int]):
        """Manually set the color of a node's mesh, was used to highlight meshes in viewer"""
        try:
            nodes = {node.name:node for node in self.node_color_map.keys()}
            self.node_color_map[nodes[node_name]] = color
        except KeyError:
            pass

    def setMaxParts(self, number_of_parts: int):
        """Limit how many meshes are rendered"""
        if number_of_parts is not None:
            self.limit_parts = True
            self.limit_number = number_of_parts
        else:
            self.limit_parts = False
        self._updateMode()

    def _updateMode(self):
        """Set the node color map depending on the type of view mode selected"""

        self.node_color_map = {}

        if self.mode == 'seg':
            if self.limit_parts:
                for joint, idx in zip(self.joint_nodes[:self.limit_number], range(self.limit_number)):
                    self.node_color_map[joint] = DEFAULT_RENDER_COLORS[idx]
            else:
                for joint, idx in zip(self.joint_nodes, range(len(self.joint_nodes))):
                    self.node_color_map[joint] = DEFAULT_RENDER_COLORS[idx]
        elif self.mode == 'seg_full':
            for joint in self.joint_nodes:
                self.node_color_map[joint] = DEFAULT_RENDER_COLORS[0]

    @property
    def resolution(self) -> Tuple[int]:
        return (self.rend.viewport_height, self.rend.viewport_width)

    @property
    def camera_pose(self) -> List[float]:
        return self.scene.get_pose(self.camera_node)

    @property
    def color_dict(self) -> dict:
        """Return a dictonary of node names and the corresponding color of the meshes"""
        if self.mode == 'seg':
            out = {}
            for node, color in zip(self.node_color_map.keys(), self.node_color_map.values()):
                out[node.name] = color
            return out
        elif self.mode == 'seg_full':
            return {'robot': DEFAULT_RENDER_COLORS[0]}



class DatasetRenderer(Renderer):
    """Renders robot poses from a dataset"""
    
    def __init__(self, dataset: str, mode: str = 'seg', camera_pose: np.ndarray = None):
        self.ds = Dataset(dataset)
        if camera_pose is None: camera_pose = self.ds.camera_pose[0]
        super().__init__(mode, camera_pose, self.ds.attrs['color_intrinsics'])
        
    def render_at(self, idx: int) -> List[np.ndarray]:
        """Render a specific index from the dataset"""
        self.setPosesFromDS(idx)
        return self.render()

    def setPosesFromDS(self, idx: int):
        """Set the robot and camera poses based on the dataset"""
        self.setJointAngles(self.ds.angles[idx])
        self.setCameraPose(self.ds.camera_pose[idx])

    def __del__(self):
        del self.ds



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

    def __init__(self, dataset: str):
        # Load dataset and corresponding renderer
        self.ds = Dataset(dataset, permissions='r+')
        self.renderer = DatasetRenderer(dataset, mode='seg_full')

        self.idx = 0
        self._findSections()
        self.section_idx = 0
        self._getSection()

        # Set value to increment index whenever going to 'next' image
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



        # Copy image array into RAM to avoid stuttering whenever moving
        print("Copying Image Array...")
        self.real_arr = np.copy(self.ds.og_img)
        self.zoom = 1

        self.gui = AlignerGUI()


    def run(self):
        """Run the Aligner"""
        ret = True
        move = True

        while ret:
            # Update the GUI
            event, values = self.gui.update(self.section_starts, self.section_idx)
            
            # If there was user input into the GUI, do what it specifies
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

            # Re-render scene if needed
            if move:
                real = self.real_arr[self.idx]
                self.renderer.setPosesFromDS(self.idx)
                render, depth = self.renderer.render()
                image = self.combineImages(real, render)
                image = self.applyZoom(image)
                move = False
            image = self.addOverlay(image)
            cv2.imshow("Aligner", image)

            # Get and enact user input
            inp = cv2.waitKey(1)
            ret, move = self.moveCamera(inp)

        # Close
        self.gui.close()
        cv2.destroyAllWindows()


    def moveCamera(self, inp: int) -> List[bool]:
        """Move the camera based on input

        Parameters
        ----------
        inp : int
            Unicode code for pressed key

        Returns
        -------
        List[bool, bool]
            [0] - If the program should continue
            [1] - If the image needs to be re-rendered
        """

        xyz_step = self.xyz_steps[self.step_loc]
        ang_step = self.ang_steps[self.step_loc]

        # Quit
        if inp == ord('0'):
            return False, False

        # Change increment scale
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

        # Change viewed image
        elif inp == ord('k'):
            self.increment(-self.inc)
            return True, True
        elif inp == ord('l'):
            self.increment(self.inc)
            return True, True

        # Move based on input
        if   inp == ord('d'):   self.c_pose[0] -= xyz_step
        elif inp == ord('a'):   self.c_pose[0] += xyz_step
        elif inp == ord('w'):   self.c_pose[1] -= xyz_step
        elif inp == ord('s'):   self.c_pose[1] += xyz_step
        elif inp == ord('z'):   self.c_pose[2] += xyz_step
        elif inp == ord('x'):   self.c_pose[2] -= xyz_step
        elif inp == ord('q'):   self.c_pose[3] -= ang_step
        elif inp == ord('e'):   self.c_pose[3] += ang_step
        elif inp == ord('r'):   self.c_pose[4] -= ang_step
        elif inp == ord('f'):   self.c_pose[4] += ang_step
        elif inp == ord('g'):   self.c_pose[5] += ang_step
        elif inp == ord('h'):   self.c_pose[5] -= ang_step

        # Save the current camera pose
        self.saveCameraPose()
        return True, True


    def applyZoom(self, image: np.ndarray) -> np.ndarray:
        """Size the image according to a 'Zoom' value"""
        dim = list(image.shape[:2])
        dim.reverse()
        dim = [x* self.zoom for x in dim]
        dim = [int(x) for x in dim]
        image = cv2.resize(image, tuple(dim))
        return image

    def addOverlay(self, image: np.ndarray) -> np.ndarray:
        """Add the overlay of the current camera pose to the image"""
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
        """Combine render and actual image into a single image"""
        return np.array(image_a * weight + image_b *(1-weight), dtype=np.uint8)


    def increment(self, step: int):
        """Change viewed pose"""
        if (self.idx + step >= 0) and (self.idx + step < self.ds.length):
            self.idx += step


    def saveCameraPose(self):
        """Set camera pose for all in the section"""
        for idx in range(self.start_idx, self.end_idx + 1):
            self.ds.camera_pose[idx,:] = self.c_pose

    def readCameraPose(self):
        """Get camera pose from ds"""
        self.c_pose = self.ds.camera_pose[self.idx,:]

    def _findSections(self) -> List[int]:
        """Find sections based on if camera pose changes"""
        self.section_starts = []
        p = [0,0,0,0,0,0]   # Dummy value
        for idx in range(self.ds.length):
            if not np.array_equal(self.ds.camera_pose[idx], p):
                self.section_starts.append(idx)
                p = self.ds.camera_pose[idx,:]

        self.section_starts.append(self.ds.length)  # Append ending value as a 'start'
        return self.section_starts

    def _newSection(self, idx: int):
        """Add a section to the section list"""
        self.section_starts.append(idx)
        self.section_starts.sort()

    def _getSection(self):
        """Set the current start/end of section"""
        section_start = max([x for x in self.section_starts if x <= self.idx])
        self.section_idx = self.section_starts.index(section_start)
        self.start_idx = section_start
        self.end_idx = self.section_starts[self.section_idx + 1] - 1



class AlignerGUI():
    """GUI that allows added control for Aligner and displays instructions"""

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

    def update(self, section_starts: List[int], section_idx: int):
        """Update the GUI, returning an evenmt if applicable"""
        
        # Read window
        event, values = self.window.read(timeout=1, timeout_key='tm')

        # Turn section starts into a table of sections
        section_table = []
        for idx in range(len(section_starts)-1):
            section_table.append([[f"{section_starts[idx]} - {section_starts[idx+1]-1}"]])

        # Display current and all sections
        self.window['editing'].update(f"{section_starts[section_idx]} - {section_starts[section_idx+1]-1}")
        self.window['sections'].update(section_table)

        # If zoom was changed, return it as the event
        if values['zoom'] != self.past_zoom:
            self.past_zoom = values['zoom']
            return ['zoom',values['zoom']]

        try:
            # Attempt to goto an index or create a new section
            if event == 'new_section':
                return ['new_section',int(values['num_input'])]
            elif event == 'num_goto':
                return ['goto',int(values['num_input'])]
            if len(values['num_input']) > 0:
                if int(values['num_input']) is not None:
                    self.window['warn'].update("")

        except ValueError:
            # If input value was not an int, warn
            self.window['warn'].update("Please input a number.")

        if event == 'quit':
            self.close()
            return ['quit',None]

        try:
            # If there is an entry in the manual pose section, try to parse and apply it
            entry = values['pose_entry'].replace('[','').replace(']','').replace(',',' ')
            entry = np.fromstring(entry,np.float,sep=' ')
            if entry.shape == (6,):
                return ['pose_entry',entry]
        except ValueError:
            pass

        return [None,None]

    def close(self):
        """Close window"""
        self.window.close()
