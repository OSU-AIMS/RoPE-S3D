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
import time

import cv2
import trimesh
import pyrender

from . import paths as p
#from .autoAnnotate import makeMask
from .dataset import Dataset
from .projection import proj_point_to_pixel, makeIntrinsics

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



def cameraFromIntrinsics(rs_intrinsics):
    """Returns Pyrender Camera.
    Makes a Pyrender camera from realsense intrinsics
    """
    return pyrender.IntrinsicsCamera(cx=rs_intrinsics.ppx, cy=rs_intrinsics.ppy, fx=rs_intrinsics.fx, fy=rs_intrinsics.fy)


def loadModels(obj_list, path = p.ROBOT_CAD, mode = 'pyrender',fileend='.obj'):
    """Returns mesh objects.
    Loads CAD models as trimesh or pyrender objects
    """
    assert mode == 'trimesh' or mode == 'pyrender'
    meshes = []
    for file in obj_list:
        if not file.endswith(fileend):
            file += fileend
        
        meshes.append(trimesh.load(os.path.join(path,file)))

    if mode == 'trimesh':
        return meshes
    else:
        out = []
        for mesh in meshes:
            out.append(pyrender.Mesh.from_trimesh(mesh,smooth=True))

        return out



def angToPoseArr(ang1,ang2,ang3, arr = None):
    """Returns 4x4 pose array.
    Converts rotations to a pose array
    """
    # Takes pitch, roll, yaw and converts into a pose arr
    angs = np.array([ang1,ang2,ang3])
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



def coordsFromData(ang, pos):
    """Returns Zx6x6 array of positions and angles.
    Given angle and positon arrays, make an array of mesh locations and rotations.
    """
    #Make arr in x,y,z,roll,pitch,yaw format
    coord = np.zeros((pos.shape[0],6,6))

    # 1:6 are movable joints, correspond to S,L,U,R and BT
    coord[:,1:6,2] = pos[:,:5,2]        # z is equal
    coord[:,1:6,0] = -1 * pos[:,:5,1]   # x = -y
    coord[:,1:6,1] = pos[:,:5,0]        # y = x

    # Yaw of all movings parts is just the S angle
    for idx in range(1,6):
        coord[:,idx,5] = ang[:,0]

    coord[:,2,4] = -1 * ang[:,1]                        # Pitch of L
    coord[:,3,4] = -1 * ang[:,1] + ang[:,2]             # Pitch of U
    coord[:,4,4] = -1 * ang[:,1] + ang[:,2] + np.pi/2   # Pitch of R
    coord[:,5,4] = -1 * ang[:,1] + ang[:,2] + ang[:,4]  # Pitch of BT

    return coord


def makePoses(coords):
    """Returns Zx6x4x4 array.
    Given a coordinate and roation array, make 4x4 pose arrays
    """
    poses = np.zeros((coords.shape[0],6,4,4))
    # X frames, 6 joints, 4x4 pose for each
    for idx in range(coords.shape[0]):
        for sub_idx in range(6):
            poses[idx,sub_idx] = makePose(*coords[idx,sub_idx])

    return poses


def setPoses(scene, nodes, poses):
    """
    Set all the poses of objects in a scene
    """
    for node, pose in zip(nodes,poses):
        scene.set_pose(node,pose)


def readCameraPose(path, idx):
    return np.load(path)[idx]




class Renderer():
    
    def __init__(
            self,
            mesh_list,
            dataset,
            skeleton,
            name_list = None,
            mode = 'seg',
            mesh_path = p.ROBOT_CAD,
            mesh_type = '.obj',
            camera_pose = None,
            camera_intrin = '1280_720_color',
            resolution = [1280, 720],
            robot_name="mh5"
            ):

        self.robot_name = robot_name

        # Load dataset
        self.ds = Dataset(dataset, skeleton)

        # Load meshes
        print("Loading Meshes")
        self.meshes = loadModels(mesh_list, mesh_path, fileend=mesh_type)
        if name_list is None:
            name_list = mesh_list

        self.cam_path = os.path.join(self.ds.dataset_dir,'camera_pose.npy')
         
        if camera_pose is not None:
            c_pose = camera_pose
        elif os.path.isfile(self.cam_path):
            c_pose = readCameraPose(self.cam_path, 0)
        else:
            c_pose = [17,0,4,0,np.pi/2,np.pi/2]    # Default Camera Pose

        self.scene = pyrender.Scene(bg_color=[0.0,0.0,0.0])  # Make scene

        camera = cameraFromIntrinsics(makeIntrinsics(camera_intrin))
        cam_pose = makePose(*c_pose)

        self.camera_node = self.scene.add(camera, pose=cam_pose)

        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=15.0)
        
        self.scene.add(dl, pose=makePose(15,0,15,0,np.pi/4,np.pi/2)) # Add light above camera
        self.scene.add(dl, pose=makePose(15,0,-15,0,3*np.pi/4,np.pi/2)) # Add light below camera
        self.scene.add(dl, pose=cam_pose) # Add light at camera pos

        # Add in joints
        self.joint_nodes = []
        for mesh,name in zip(self.meshes, name_list):
            self.joint_nodes.append(pyrender.Node(name=name,mesh=mesh))

        for node in self.joint_nodes:
            self.scene.add_node(node)

        # Add in keypoint markers
        self.key_nodes = []
        marker = trimesh.creation.cylinder(
            self.ds.keypoint_data['markers']['radius'],
            height=self.ds.keypoint_data['markers']['height']
            )
        marker = pyrender.Mesh.from_trimesh(marker)

        for name in self.ds.keypoint_data['keypoints'].keys():
            parent = self.ds.keypoint_data['keypoints'][name]['parent_joint']
            pose = makePose(*self.ds.keypoint_data['keypoints'][name]['pose'])
            n = self.scene.add(marker, name=name, pose=pose, parent_name=parent)
            self.key_nodes.append(n)


        self.rend = pyrender.OffscreenRenderer(*resolution)

        self.setMode(mode)


    def render(self):
        return self.rend.render(
            self.scene,
            flags=pyrender.constants.RenderFlags.SEG,
            seg_node_map=self.node_color_map
            )



    def setObjectPoses(self, poses):
        setPoses(self.scene, self.joint_nodes, poses)


    def setPosesFromDS(self, idx):
        if not hasattr(self,'ds_poses'):
            self.ds_poses = makePoses(coordsFromData(self.ds.angles, self.ds.positions))
        self.setObjectPoses(self.ds_poses[idx])

        setPoses(self.scene, [self.camera_node], [makePose(*readCameraPose(self.cam_path,idx))])


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


    def setMode(self, mode):
        valid_modes = ['seg','key','seg_full']
        assert mode in valid_modes, f"Mode invalid; must be one of: {valid_modes}"

        self.mode = mode

        self.node_color_map = {}

        if mode == 'seg':
            for joint, idx in zip(self.joint_nodes, range(len(self.joint_nodes))):
                self.node_color_map[joint] = DEFAULT_COLORS[idx]
            
        elif mode == 'key':
            for keypt, idx in zip(self.key_nodes, range(len(self.key_nodes))):
                self.node_color_map[keypt] = DEFAULT_COLORS[idx]
            for joint in self.joint_nodes:
                self.node_color_map[joint] = DEFAULT_COLORS[-1]

        elif mode == 'seg_full':
            for joint in self.joint_nodes:
                self.node_color_map[joint] = DEFAULT_COLORS[0]
 



class Aligner():
    """
    Used to manually find the position of camera relative to robot.
    """
    """
    W/S - Move forward/backward
    A/D - Move left/right
    Z/X - Move down/up
    Q/E - Roll
    R/F - Tilt down/up
    G/H - Pan left/right
    +/- - Increase/Decrease Step size
    """

    def __init__(
            self,
            mesh_list,
            name_list,
            dataset,
            skeleton,
            start_idx = None,
            end_idx = None
            ):
        # Load dataset
        self.ds = Dataset(dataset, skeleton)

        self.renderer = Renderer(mesh_list, dataset, skeleton, mode='seg_full',name_list=name_list)
        self.cam_path = self.renderer.cam_path


        # Image counter
        self.idx = 0
        if start_idx is not None:
            self.start_idx = start_idx
        else:
            self.start_idx = 0

        if end_idx is not None:
            self.end_idx = end_idx
        else:
            self.end_idx = self.ds.length - 1


        # Read in camera pose if it's been written before
        if os.path.isfile(self.cam_path):
            self.readCameraPose(self.start_idx)
        else:
            # Init pose, write
            self.c_pose = [1.425,.009,0.416,-.01,np.pi/2,np.pi/2]
            a = np.zeros((self.ds.length,6))
            for idx in range(self.ds.length):
                a[idx] = self.c_pose
            np.save(self.cam_path,a)


        # Movement steps
        self.xyz_steps = [.001,.005,.01,.05,.1,.25,.5]
        self.ang_steps = [.0005,.001,.005,.01,.025,.05,.1]
        self.step_loc = len(self.xyz_steps) - 4


    def setCameraPose(self):
        self.camera_pose_arr = makePose(*self.c_pose)
        self.renderer.scene.set_pose(self.renderer.camera_node, self.camera_pose_arr)


    def run(self):
        ret = True

        while ret:

            self.setCameraPose()
            real = self.ds.og_img[self.idx]
            self.renderer.setPosesFromDS(self.idx)
            render, depth = self.renderer.render()
            image = self.combineImages(real, render)
            image = self.addOverlay(image)
            cv2.imshow("Aligner", image)

            inp = cv2.waitKey(0)
            ret = self.moveCamera(inp)

        cv2.destroyAllWindows()



    def moveCamera(self,inp):
        """
        W/S - Move forward/backward
        A/D - Move left/right
        Z/X - Move up/down
        Q/E - Roll
        R/F - Tilt down/up
        G/H - Pan left/right
        +/- - Increase/Decrease Step size
        K/L - Last/Next image
        0 - Quit
        """

        xyz_step = self.xyz_steps[self.step_loc]
        ang_step = self.ang_steps[self.step_loc]

        if inp == ord('0'):
            return False
        elif inp == ord('='):
            self.step_loc += 1
            if self.step_loc >= len(self.xyz_steps):
                self.step_loc = len(self.xyz_steps) - 1
            return True
        elif inp == ord('-'):
            self.step_loc -= 1
            if self.step_loc < 0:
                self.step_loc = 0
            return True
        elif inp == ord('k'):
            self.increment(-5)
            return True
        elif inp == ord('l'):
            self.increment(5)
            return True

        if inp == ord('w'):
            self.c_pose[0] -= xyz_step
        elif inp == ord('s'):
            self.c_pose[0] += xyz_step
        elif inp == ord('a'):
            self.c_pose[1] -= xyz_step
        elif inp == ord('d'):
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
        return True


    def addOverlay(self, image):
        pose_str = "["
        for num in self.c_pose:
            pose_str += f"{num:.3f}, "
        pose_str +="]"
        image = cv2.putText(image, pose_str,(10,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        image = cv2.putText(image, str(self.xyz_steps[self.step_loc]),(10,100), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        image = cv2.putText(image, str(self.ang_steps[self.step_loc]),(10,150), cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        return image


    def combineImages(self,image_a, image_b, weight = .5):
        return np.array(image_a * weight + image_b *(1-weight), dtype=np.uint8)


    def increment(self, step):
        if not (self.idx + step >= self.end_idx) and not (self.idx + step < self.start_idx):
            self.idx += step


    def saveCameraPose(self):
        arr = np.load(self.cam_path)
        for idx in range(self.start_idx, self.end_idx + 1):
            arr[idx] = self.c_pose
        while True:
            try:
                np.save(self.cam_path, arr)
                break
            except PermissionError:
                time.sleep(.05)

    def readCameraPose(self,idx):
        self.c_pose = np.load(self.cam_path)[idx]
