import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import cv2
from .projection import makeIntrinsics
import os
from . import paths as p


def cameraFromIntrinsics(rs_intrinsics):
    return pyrender.IntrinsicsCamera(cx=rs_intrinsics.ppx, cy=rs_intrinsics.ppy, fx=rs_intrinsics.fx, fy=rs_intrinsics.fy)


def loadOBJs(obj_list, path = p.robot_cad, mode = 'pyrender'):
    assert mode == 'trimesh' or mode == 'pyrender'
    meshes = []
    for file in obj_list:
        if not file.endswith('.obj'):
            file += '.obj'
        
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




def test_render():

    objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS_NEW','MH5_BT_UNIFIED_AXIS']
    meshes = loadOBJs(objs)
    coords = loadCoords()
    poses = makePoses(coords)

    test_pose = poses[60]

    scene = pyrender.Scene()
    num_of_joints_to_do = 6

    nodes = []
    for mesh, pose in zip(meshes[0:num_of_joints_to_do],test_pose[0:num_of_joints_to_do]):
        nodes.append(scene.add(mesh, pose=pose))



    camera = cameraFromIntrinsics(makeIntrinsics())
    s = np.sqrt(2)/2
    camera_pose = makePose(17,0,4,0,np.pi/2,np.pi/2) # X,Y,Z, Roll(+CW,CCW-), Tilt(+Up,Down-), Pan(+Left,Right-) 

    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=1000.0,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)
    dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=15.0)
    scene.add(dl, pose=camera_pose)
    r = pyrender.OffscreenRenderer(1280, 720)


    for frame in range(100):
        frame_poses = poses[frame,:num_of_joints_to_do]
        setPoses(scene, nodes, frame_poses)
        color, depth = r.render(scene)
        cv2.imshow("Render", color) 
        cv2.waitKey(1)




if __name__ == "__main__":
    test_render()