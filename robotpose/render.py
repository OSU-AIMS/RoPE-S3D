# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley


import numpy as np
import os
import json

import cv2
import pyrender
import PySimpleGUI as sg
import trimesh

from . import paths as p
from .dataset import Dataset
from .projection import makeIntrinsics

MESH_CONFIG = os.path.join(p.DATASETS,'mesh_config.json')

DEFAULT_COLORS = [
    [0  , 0  , 85 ],[0  , 0  , 170],[0  , 0  , 255],
    [0  , 85 , 0  ],[0  , 170, 0  ],[0  , 255, 0  ],
    [85 , 0  , 0  ],[170, 0  , 0  ],[255, 0  , 0  ],
    [0  , 0  , 85 ],[0  , 85 , 85 ],[85 , 0  , 85 ],[85 , 85  , 0 ],
    [0  , 0  , 170],[0  , 170, 170],[170, 0  , 170],[170, 170 , 0 ],
    [0  , 0  , 255],[0  , 255, 255],[255, 0  , 255],[255, 255 , 0 ],
    [170, 85 , 85 ],[85 , 170, 85 ],[85 , 85 , 170],
    [255, 85 , 85 ],[85 , 255, 85 ],[85 , 85 , 255],
    [255, 170, 170],[170, 255, 170],[170, 170, 255],
    [85 , 170, 170],[170, 85 , 170],[170, 170, 85 ],
    [85 , 255, 255],[255, 85 , 255],[255, 255, 85 ],
    [85 , 170, 255],[255, 85 , 170],[170, 255, 85 ],[255, 85 , 170],[170, 255, 85 ],
    [85 , 85 , 85]
]



class MeshLoader():

    def __init__(self, mesh_dir = p.ROBOT_CAD):
        self.mesh_dir = mesh_dir

        if not os.path.isfile(MESH_CONFIG):
            print("\nWARNING: No mesh config present. Making default.")
            info = {}
            default_pose = [0,0,0,0,0,0]
            joints = ['BASE','S','L','U','R','BT']
            for joint in joints:
                info[joint] = {"file_name":f"{joint}.obj","pose":default_pose}
            with open(MESH_CONFIG,'w') as f:
                json.dump(info, f, indent=4)
            raise ValueError(f"\n\nMesh config file was not present. Please edit {MESH_CONFIG} to be accurate.")

        with open(MESH_CONFIG,'r') as f:
            d = json.load(f)

        self.name_list = [x for x in d.keys()]
        self.mesh_list = [d[x]['file_name'] for x in self.name_list]
        self.pose_list = [d[x]['pose'] for x in self.name_list]

    def load(self):
        self.meshes = []
        for file, pose in zip(self.mesh_list, self.pose_list):
            tm = trimesh.load(os.path.join(self.mesh_dir,file))
            self.meshes.append(pyrender.Mesh.from_trimesh(tm,smooth=True, poses=makePose(*pose)))

    def getMeshes(self):
        return self.meshes

    def getNames(self):
        return self.name_list



def cameraFromIntrinsics(rs_intrinsics):
    """Returns Pyrender Camera.
    Makes a Pyrender camera from realsense intrinsics
    """
    return pyrender.IntrinsicsCamera(cx=rs_intrinsics.ppx, cy=rs_intrinsics.ppy, fx=rs_intrinsics.fx, fy=rs_intrinsics.fy)


def angToPoseArr(yaw,pitch,roll, arr = None):
    """Returns 4x4 pose array.
    Converts rotations to a pose array
    """
    # Takes pitch, roll, yaw and converts into a pose arr
    angs = np.array([yaw,pitch,roll])
    c = np.cos(angs)
    s = np.sin(angs)
    if arr is None:
        pose = np.zeros((4,4))
    else:
        pose = arr

    pose[0,0] = c[0] * c[1]
    pose[1,0] = c[1] * s[0]
    pose[2,0] = -1 * s[1]

    pose[0,1] = c[0] * s[1] * s[2] - c[2] * s[0]
    pose[1,1] = c[0] * c[2] + np.prod(s)
    pose[2,1] = c[1] * s[2]

    pose[0,2] = s[0] * s[2] + c[0] * c[2] * s[1]
    pose[1,2] = c[2] * s[0] * s[1] - c[0] * s[2]
    pose[2,2] = c[1] * c[2]

    pose[3,3] = 1.0

    return pose
    

def translatePoseArr(x,y,z, arr = None):
    """Returns 4x4 pose array.
    Translates a pose array
    """
    if arr is None:
        pose = np.zeros((4,4))
    else:
        pose = arr

    pose[0,3] = x
    pose[1,3] = y
    pose[2,3] = z

    return pose


def makePose(x,y,z,pitch,roll,yaw):
    """Returns 4x4 pose array.
    Makes pose array given positon and angle
    """
    pose = angToPoseArr(yaw,pitch,roll)
    pose = translatePoseArr(x,y,z,pose)
    return pose



def posesFromData(ang, pos):
    """
    Returns Zx6x4x4 array of pose arrays
    """
    #Make arr in x,y,z,roll,pitch,yaw format
    coord = np.zeros((pos.shape[0],6,6))

    # Yaw of S-R is just S angle
    for idx in range(1,5):
        coord[:,idx,5] = ang[:,0]

    coord[:,1:6,0:3] = pos[:,:5]    # Set xyz translations

    coord[:,2,3] = ang[:,1]             # Pitch of L
    coord[:,3,3] = ang[:,1] - ang[:,2]  # Pitch of U
    coord[:,4,3] = ang[:,1] - ang[:,2]  # Pitch of R       

    coord[:,4,4] = -ang[:,3]    # Roll of R


    poses = np.zeros((coord.shape[0],6,4,4))    # X frames, 6 joints, 4x4 pose for each
    for idx in range(coord.shape[0]):
        for sub_idx in range(5):
            poses[idx,sub_idx] = makePose(*coord[idx,sub_idx])

    # Determine BT with vectors because it's easier

    bt = pos[:,5] - pos[:,4]    # Vectors from B to T

    y = poses[:,4,:3,1] # Y Axis is common with R joint
    z = np.cross(bt, y)

    z = z/np.linalg.norm(z)
    x = bt/np.linalg.norm(bt)

    b_poses = np.zeros((pos.shape[0],4,4))

    b_poses[:,:3,0] = x # X Unit
    b_poses[:,:3,1] = y # Y Unit
    b_poses[:,:3,2] = z # Z Unit
    b_poses[:,3,3] = 1
    b_poses[:,:3,3] = pos[:,4] # XYZ Offset



    poses[:,-1] = b_poses

    return poses



def setPoses(scene, nodes, poses):
    """
    Set all the poses of objects in a scene
    """
    for node, pose in zip(nodes,poses):
        scene.set_pose(node,pose)





class Renderer():
    
    def __init__(
            self,
            dataset,
            skeleton = None,
            ds_type = 'full',
            mode = 'seg',
            camera_pose = None,
            camera_intrin = '1280_720_color',
            resolution = [1280, 720],
            robot_name="mh5"
            ):

        self.mode = mode

        self.robot_name = robot_name
        self.skeleton = skeleton
        # Load dataset
        self.ds = Dataset(dataset, skeleton, ds_type = ds_type)

        # Load meshes
        ml = MeshLoader()
        ml.load()
        name_list = ml.getNames()
        self.meshes = ml.getMeshes()
         
        if camera_pose is not None:
            c_pose = camera_pose
        else:
            c_pose = self.ds.camera_pose[0]

        self.scene = pyrender.Scene(bg_color=[0.0,0.0,0.0])  # Make scene

        camera = cameraFromIntrinsics(makeIntrinsics(camera_intrin))
        cam_pose = makePose(*c_pose)
        self.camera_node = self.scene.add(camera, pose=cam_pose)

        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=15.0)
        self.scene.add(dl, parent_node=self.camera_node) # Add light at camera pos

        # Add in joints
        self.joint_nodes = []
        for mesh,name in zip(self.meshes, name_list):
            self.joint_nodes.append(pyrender.Node(name=name,mesh=mesh))

        for node in self.joint_nodes:
            self.scene.add_node(node)

        if self.skeleton is not None:
            self._updateKeypoints()

        self.rend = pyrender.OffscreenRenderer(*resolution)

        self.setMode(mode)


    def render(self, update_keypoints = False):
        if update_keypoints and self.skeleton is not None:
            self.ds.updateKeypointData()
            self._updateKeypoints()
        return self.rend.render(
            self.scene,
            flags=pyrender.constants.RenderFlags.SEG,
            seg_node_map=self.node_color_map
            )


    def setMode(self, mode):
        valid_modes = ['seg','key','seg_full']
        assert mode in valid_modes, f"Mode invalid; must be one of: {valid_modes}"
        if self.skeleton is None:
            assert mode != 'key', f"Skeleton must be specified for keypoint rendering"
        self.mode = mode
        self._updateMode()


    def setObjectPoses(self, poses):
        setPoses(self.scene, self.joint_nodes, poses)


    def setPosesFromDS(self, idx):
        if not hasattr(self,'ds_poses'):
            self.ds_poses = posesFromData(self.ds.angles, self.ds.positions)
        self.setObjectPoses(self.ds_poses[idx])

        setPoses(self.scene, [self.camera_node], [makePose(*self.ds.camera_pose[idx])])


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
            return {self.robot_name: DEFAULT_COLORS[0]}


    def _updateKeypoints(self):
        # Remove olds
        if hasattr(self, 'key_nodes'):
            if len(self.key_nodes) > 0:
                for node in self.key_nodes:
                    if self.scene.has_node(node):
                        self.scene.remove_node(node)

        # Add in new
        self.key_nodes = []
        marker = trimesh.creation.cylinder(
            self.ds.skele.data['markers']['radius'],
            height=self.ds.skele.data['markers']['height']
            )
        marker = pyrender.Mesh.from_trimesh(marker)

        for name in self.ds.skele.keypoints:
            parent = self.ds.skele.keypoint_data[name]['parent_joint']
            pose = makePose(*self.ds.skele.keypoint_data[name]['pose'])
            n = self.scene.add(marker, name=name, pose=pose, parent_name=parent)
            self.key_nodes.append(n)

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

        self.renderer = Renderer(dataset, None, mode='seg_full')

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

            self._getSection()
            self.readCameraPose()

            if move:
                real = self.real_arr[self.idx]
                self.renderer.setPosesFromDS(self.idx)
                render, depth = self.renderer.render()
                image = self.combineImages(real, render)
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
        self.section_starts

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
                        [sg.Text("",key='warn',text_color="red", size=(22,1))],
                        [sg.Table([[["Sections:"]],[[1,1]]], key='sections'),sg.Text(control_str)],
                        [sg.Button('Quit',key='quit')]]

        self.window = sg.Window('Aligner Controls', self.layout, return_keyboard_events = True, use_default_focus=False)

    def update(self, section_starts, section_idx):
        event, values = self.window.read(timeout=1, timeout_key='tm')
        section_table = []
        for idx in range(len(section_starts)-1):
            section_table.append([[f"{section_starts[idx]} - {section_starts[idx+1]-1}"]])

        self.window['editing'].update(f"{section_starts[section_idx]} - {section_starts[section_idx+1]-1}")
        self.window['sections'].update(section_table)

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

        return [None,None]

    def close(self):
        self.window.close()
