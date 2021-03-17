# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import numpy as np
from numpy.core.numeric import full
from pyrender import renderer
import trimesh
import pyrender
import matplotlib.pyplot as plt
import cv2
from .projection import makeIntrinsics
import os
from . import paths as p
from .autoAnnotate import Annotator, labelSegmentation, makeMask
import json
from .dataset import Dataset
from .turbo_colormap import normalize_and_interpolate
from .projection import proj_point_to_pixel, makeIntrinsics


from .utils import outlier_min_max
import random

def cameraFromIntrinsics(rs_intrinsics):
    return pyrender.IntrinsicsCamera(cx=rs_intrinsics.ppx, cy=rs_intrinsics.ppy, fx=rs_intrinsics.fx, fy=rs_intrinsics.fy)


def loadModels(obj_list, path = p.robot_cad, mode = 'pyrender',fileend='.obj'):
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
    if arr is None:
        pose = np.zeros((4,4))
    else:
        pose = arr

    pose[0,3] = x
    pose[1,3] = y
    pose[2,3] = z

    return pose


def makePose(x,y,z,pitch,roll,yaw):
    pose = angToPoseArr(yaw,pitch,roll)
    pose = translatePoseArr(x,y,z,pose)
    return pose


def loadCoords():
    pos_path = r'data/set6_slu/pos.npy'
    ang_path = r'data/set6_slu/ang.npy'

    pos = np.load(pos_path)
    ang = np.load(ang_path)
    assert pos.shape[0] == ang.shape[0]

    #Make arr in x,y,z,roll,pitch,yaw format
    coord = np.zeros((pos.shape[0],6,6))

    # 1:6 are movable joints, correspond to S,L,U,R and BT
    coord[:,1:6,2] = pos[:,:5,2] # z is equal
    coord[:,1:6,0] = -1 * pos[:,:5,1] # x = -y
    coord[:,1:6,1] = pos[:,:5,0] # y = x

    # I'm dumb so these all use dm instead of m
    coord[:,:,0:3] *= 10

    # Yaw of all movings parts is just the S angle
    for idx in range(1,6):
        coord[:,idx,5] = ang[:,0]

    # Pitch of L
    coord[:,2,4] = -1 * ang[:,1]

    # Pitch of U
    coord[:,3,4] = -1 * ang[:,1] + ang[:,2]

    # Pitch of R?
    coord[:,4,4] = -1 * ang[:,1] + ang[:,2] + np.pi/2

    # Pitch of BT
    coord[:,5,4] = -1 * ang[:,1] + ang[:,2] + ang[:,4]

    return coord


def makePoses(coords):
    poses = np.zeros((coords.shape[0],6,4,4))
    # X frames, 6 joints, 4x4 pose for each
    for idx in range(coords.shape[0]):
        for sub_idx in range(6):
            poses[idx,sub_idx] = makePose(*coords[idx,sub_idx])

    return poses


def setPoses(scene, nodes, poses):
    for node, pose in zip(nodes,poses):
        scene.set_pose(node,pose)


def readCameraPose(path):
    with open(path,'r') as f:
        d = json.load(f)
    return d['pose']




def test_render():

    objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS_NEW','MH5_BT_UNIFIED_AXIS']
    #stls = ['base_link', 'link_s','link_l','link_u','link_r','link_b']
    meshes = loadModels(objs)
    #meshes = loadModels(stls,fileend='.stl')
    coords = loadCoords()
    poses = makePoses(coords)

    test_pose = poses[60]

    scene = pyrender.Scene(bg_color=[0.0,0.0,0.0])


    node_map = dict()
    nodes = []
    label_dict = dict()
    # full_c = [55,138,243]
    # label_dict["mh5"] = full_c
    for mesh, name in zip(meshes,objs):
        #mesh.primitives[0].material = pyrender.MetallicRoughnessMaterial(metallicFactor=0)
        n = scene.add(mesh)
        nodes.append(n)
        node_map[n] = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
        label_dict[name] = node_map[n]
        # node_map[n] = full_c




    camera = cameraFromIntrinsics(makeIntrinsics())
    #c_pose = [17,0,4,0,np.pi/2,np.pi/2]
    c_pose = readCameraPose(r'data/set6_slu/camera_pose.json')
    camera_pose = makePose(*c_pose) # X,Y,Z, Roll(+CW,CCW-), Tilt(+Up,Down-), Pan(+Left,Right-) 

    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=1000.0,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)
    dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=15.0)
    
    scene.add(dl, pose=makePose(15,0,15,0,np.pi/4,np.pi/2)) # Add light above camera
    scene.add(dl, pose=makePose(15,0,-15,0,3*np.pi/4,np.pi/2)) # Add light below camera
    scene.add(dl, pose=camera_pose) # Add light at camera pos
    r = pyrender.OffscreenRenderer(1280, 720)

    anno = Annotator(label_dict)

    imgs = np.load(r'data/set6_slu/og_img.npy')

    for frame in range(100):
        frame_poses = poses[frame]
        setPoses(scene, nodes, frame_poses)
        color, depth = r.render(scene,flags=pyrender.constants.RenderFlags.SEG,seg_node_map=node_map)
        print(f"{np.min(depth)},{np.max(depth)}")
        cv2.imshow("Render", color) 
        #cv2.imwrite(fr'seg_test/{frame}.png', color)
        #labelSegmentation(fr'seg_test/{frame:3d}.png',color)
        #labelSegmentation(color,color,fr'seg_test/{frame:03d}')
        anno.annotate(imgs[frame],color,fr'seg_test/{frame:03d}')
        
        cv2.waitKey(1)







class Aligner():
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

        # setup
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



import matplotlib.pyplot as plt

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
    #mn,mx = outlier_min_max(diff_vec[np.where(np.all(diff_vec != 0.0, axis=-1))])
    for r in range(depth.shape[0]):
        for c in [x for x in range(depth.shape[1]) if mask[r,x]]:
            out[r,c] = normalize_and_interpolate(diff[r,c], -.5, .5)

    return out

