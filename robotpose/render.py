# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import json
import numpy as np
import os

import cv2
import trimesh
import pyrender

from . import paths as p
from .autoAnnotate import KeypointAnnotator, SegmentationAnnotator, makeMask
from .dataset import Dataset
from .projection import proj_point_to_pixel, makeIntrinsics
from .turbo_colormap import normalize_and_interpolate


DEFAULT_COLORS = [
    [0  , 0  , 255],    # Red
    [0  , 125, 255],    # Orange
    [0  , 255, 0  ],    # Green
    [255, 255, 0  ],    # Cyan
    [255, 0  , 0  ],    # Blue
    [255, 0  , 125],    # Purple
    [255, 0  , 255],    # Pink
    [125, 0  , 255]     # Fuchsia
]


def cameraFromIntrinsics(rs_intrinsics):
    """Returns Pyrender Camera.
    Makes a Pyrender camera from realsense intrinsics
    """
    return pyrender.IntrinsicsCamera(cx=rs_intrinsics.ppx, cy=rs_intrinsics.ppy, fx=rs_intrinsics.fx, fy=rs_intrinsics.fy)


def loadModels(obj_list, path = p.robot_cad, mode = 'pyrender',fileend='.obj'):
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

    # I'm dumb so these all use dm instead of m
    coord[:,:,0:3] *= 10

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


def readCameraPose(path):
    with open(path,'r') as f:
        d = json.load(f)
    return d['pose']





def test_render_with_class():

    objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS_NEW','MH5_BT_UNIFIED_AXIS']
    names = ['BASE','S','L','U','R','BT']

    r = Renderer(objs, name_list=names)
    r.setMode('key')
    color_dict = r.getColorDict()
    anno = KeypointAnnotator(color_dict,'set6','B')
    

    for frame in range(100):
            
        r.setPosesFromDS(frame)
        color,depth = r.render()
        anno.annotate(color,frame)
        cv2.imshow("Render", color)
        cv2.waitKey(100)





class Renderer():
    
    def __init__(
            self,
            mesh_list,
            name_list = None,
            mode = 'seg',
            mesh_path = p.robot_cad,
            mesh_type = '.obj',
            dataset='set6',
            skeleton='B',
            camera_pose = None,
            camera_intrin = '1280_720_color',
            resolution = [1280, 720]
            ):

        # Load dataset
        self.ds = Dataset(dataset, skeleton, load_seg=False, load_og=False, load_ply=False)

        # Load meshes
        print("Loading Meshes")
        self.meshes = loadModels(mesh_list, mesh_path, fileend=mesh_type)
        if name_list is None:
            name_list = mesh_list

        cam_path = os.path.join(self.ds.path,'camera_pose.json')
         
        if camera_pose is not None:
            c_pose = camera_pose
        elif os.path.isfile(cam_path):
            c_pose = readCameraPose(cam_path)
        else:
            c_pose = [17,0,4,0,np.pi/2,np.pi/2]    # Default Camera Pose

        self.scene = pyrender.Scene(bg_color=[0.0,0.0,0.0])  # Make scene

        camera = cameraFromIntrinsics(makeIntrinsics(camera_intrin))
        cam_pose = makePose(*c_pose)

        self.scene.add(camera, pose=cam_pose)

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
            self.ds_poses = makePoses(coordsFromData(self.ds.ang, self.ds.pos))
        self.setObjectPoses(self.ds_poses[idx])


    def getColorDict(self):
        out = {}
        for node, color in zip(self.node_color_map.keys(), self.node_color_map.values()):
            out[node.name] = color
        return out


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

    def __init__(self,dataset='set6',skeleton='B'):
        # Load dataset
        self.ds = Dataset(dataset,skeleton,load_seg= False, load_og=True, primary="og")

        self.cam_path = os.path.join(self.ds.path,'camera_pose.json')

        # Read in camera pose if it's been written before
        if os.path.isfile(self.cam_path):
            self.readCameraPose()
        else:
            # Init pose, write
            self.c_pose = [17,0,4,0,np.pi/2,np.pi/2]
            self.saveCameraPose()

        self.scene = pyrender.Scene(bg_color=[0.0,0.0,0.0])
        self.renderer = pyrender.OffscreenRenderer(1280, 720)

        # Load meshes and poses
        objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS_NEW','MH5_BT_UNIFIED_AXIS']
        self.meshes = loadModels(objs)
        coords = self.loadCoords()
        self.poses = makePoses(coords)

        # Put items into scene
        self.nodes = []
        for mesh, pose in zip(self.meshes,self.poses[0]):
            mesh.primitives[0].material = pyrender.MetallicRoughnessMaterial(metallicFactor=0)
            self.nodes.append(self.scene.add(mesh, pose=pose))

        # Make camera
        camera = cameraFromIntrinsics(makeIntrinsics())
        dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=15.0)

        # Add in camera/light
        self.camera_pose_arr = makePose(*self.c_pose) # X,Y,Z, Roll(+CW,CCW-), Tilt(+Up,Down-), Pan(+Left,Right-) 
        self.cam = self.scene.add(camera, pose=self.camera_pose_arr)
        self.light = self.scene.add(dl, pose=self.camera_pose_arr) # Add light at camera pos

        self.scene.add(dl, pose=makePose(15,0,15,0,np.pi/4,np.pi/2)) # Add light above robot
        self.scene.add(dl, pose=makePose(15,0,-15,0,3*np.pi/4,np.pi/2)) # Add light below robot

        # Image counter
        self.idx = 0

        # Movement steps
        self.xyz_steps = [.01,.05,.1,.25,.5,1,5,10]
        self.ang_steps = [.005,.01,.025,.05,.1,.25,.5,1]
        self.step_loc = len(self.xyz_steps) - 4



    def run(self):
        ret = True

        while ret:

            self.camera_pose_arr = makePose(*self.c_pose)
            setPoses(self.scene,[self.cam, self.light],[self.camera_pose_arr,self.camera_pose_arr])
            real = self.ds.img[self.idx]
            self.updatePoses()
            render, depth = self.renderFrame(do_depth=True)
            image = self.combineImages(real, render)
            image = self.addOverlay(image)
            cv2.imshow("Aligner", image)

            depth_cmp = compare_depth(self.ds.ply[self.idx], render, depth)
            cv2.imshow("Aligner_depth", depth_cmp)


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
        elif inp == ord('='):
            self.step_loc += 1
            if self.step_loc >= len(self.xyz_steps):
                self.step_loc = len(self.xyz_steps) - 1
        elif inp == ord('-'):
            self.step_loc -= 1
            if self.step_loc < 0:
                self.step_loc = 0
        elif inp == ord('k'):
            self.increment(-5)
        elif inp == ord('l'):
            self.increment(5)

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
        #return cv2.addWeighted(image_a,.4,image_b,.1,0)


    def increment(self, step):
        if not (self.idx + step >= self.ds.length) and not (self.idx + step < 0):
            self.idx += step


    def renderFrame(self, do_depth = False):
        color, depth = self.renderer.render(self.scene)
        if do_depth:
            return color, depth
        else:
            return color


    def saveCameraPose(self):
        with open(self.cam_path,'w') as f:
            json.dump({'pose':self.c_pose},f)

    def readCameraPose(self):
        with open(self.cam_path,'r') as f:
            d = json.load(f)
        self.c_pose = d['pose']


    def updatePoses(self):
        frame_poses = self.poses[self.idx]
        setPoses(self.scene, self.nodes, frame_poses)


    def loadCoords(self):
        pos = self.ds.pos
        ang = self.ds.ang
        assert pos.shape[0] == ang.shape[0]

        coord = np.zeros((pos.shape[0],6,6))

        coord[:,1:6,2] = pos[:,:5,2] # z is equal
        coord[:,1:6,0] = -1 * pos[:,:5,1] # x = -y
        coord[:,1:6,1] = pos[:,:5,0] # y = x

        # I'm dumb so these all use dm instead of m
        coord[:,:,0:3] *= 10
        for idx in range(1,6):
            coord[:,idx,5] = ang[:,0]
        coord[:,2,4] = -1 * ang[:,1]
        coord[:,3,4] = -1 * ang[:,1] + ang[:,2]
        coord[:,4,4] = -1 * ang[:,1] + ang[:,2] + np.pi/2
        coord[:,5,4] = -1 * ang[:,1] + ang[:,2] + ang[:,4]

        return coord




def compare_depth(ply, color, depth, ply_multiplier = -10):
    ply_frame_data = np.copy(ply)
    mask = makeMask(color)
    ply_depth = np.zeros(depth.shape)

    # Reproject pixels
    intrin = makeIntrinsics()
    ply_frame_data[:,0:2] = proj_point_to_pixel(intrin,ply_frame_data[:,2:5])
    idx_arr = ply_frame_data[:,0:2].astype(int)
    idx_arr[:,0] = np.clip(idx_arr[:,0],0,1279)
    idx_arr[:,1] = np.clip(idx_arr[:,1],0,719)
    for idx in range(len(ply_frame_data)):
        ply_depth[idx_arr[idx,1],idx_arr[idx,0]] = ply_multiplier * ply_frame_data[idx,4]


    diff_vec = np.subtract(depth,ply_depth)
    diff = np.copy(diff_vec)
    diff_vec.flatten()

    # plt.hist(diff_vec)
    # plt.show()

    out = np.zeros((depth.shape[0], depth.shape[1],3), np.uint8)
    for r in range(depth.shape[0]):
        for c in [x for x in range(depth.shape[1]) if mask[r,x]]:
            out[r,c] = normalize_and_interpolate(diff[r,c], -.5, .5)

    return out

