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

    pose[0,0] = c[1] * c[2]
    pose[1,0] = c[0] * s[2] + c[2] * s[0] * s[1]
    pose[2,0] = s[0] * s[2] - c[0] * c[2] * s[1]

    pose[0,1] = -1 * c[1] * s[2]
    pose[1,1] = c[0] * c[2] - np.prod(s)
    pose[2,1] = c[2] * s[0] + c[0] * s[1] * s[2]

    pose[0,2] = s[1]
    pose[1,2] = -1 * c[1] * s[0]
    pose[2,2] = c[0] * c[1]

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
    pose = angToPoseArr(pitch,roll,yaw)
    pose = translatePoseArr(x,y,z,pose)
    return pose


def loadCoords():
    pos_path = r'data/set6_slu/pos.npy'
    ang_path = r'data/set6_slu/ang.npy'

    pos = np.load(pos_path)
    ang = np.load(ang_path)
    assert pos.shape[0] == ang.shape[0]

    #Make arr in x,y,z,pitch,roll,yaw format

    coord = np.zeros((pos.shape[0],6,6))
    coord[:,:,0:3] = pos

    return coord


def makePoses(coords):
    poses = np.zeros((coords.shape[0],6,4,4))
    # X frames, 6 joints, 4x4 pose for each
    for idx in range(coords.shape[0]):
        for sub_idx in range(6):
            poses[idx,sub_idx] = makePose(*coords[idx,sub_idx])

    return poses



def test_render():

    objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS','MH5_BT_UNIFIED_AXIS']
    meshes = loadOBJs(objs)
    coords = loadCoords()
    poses = makePoses(coords)

    test_pose = poses[50]

    scene = pyrender.Scene()
    num_of_joints_to_do = 3

    for mesh, pose in zip(meshes[0:num_of_joints_to_do],test_pose[0:num_of_joints_to_do]):
        print(pose)
        scene.add(mesh, pose=pose)


    # scene.add(mesha)
    # scene.add(meshb)
    # scene.add(meshc)
    # scene.add(meshd)
    #camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)

    #intrin((1280,720), (638.391,361.493), (905.23, 904.858), rs.distortion.inverse_brown_conrady, [0,0,0,0,0])
    camera = cameraFromIntrinsics(makeIntrinsics())
    #s = np.sqrt(2)/2
    s = np.sqrt(2)/2
    camera_pose = np.array([
        [0.0, -s,   s,   5],
        [1.0,  0.0, 0.0, 0.0],
        [0.0,  s,   s,   2],
        [0.0,  0.0, 0.0, 1.0],
     ])

    a = 0
    b = .9

    camera_pose = np.array([
        [0.0,               -a,                 b,   20],
        [1.0,              0.0,               0.0, 0.0],
        [0.0,  np.sqrt(1-a**2),   np.sqrt(1-b**2),   8],
        [0.0,               0.0,              0.0, 1.0],
     ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=1000.0,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)
    dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=15.0)
    scene.add(dl, pose=camera_pose)
    r = pyrender.OffscreenRenderer(1280, 720)
    color, depth = r.render(scene)
    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.axis('off')
    # plt.imshow(color)
    # plt.subplot(1,2,2)
    # plt.axis('off')
    # plt.imshow(depth, cmap=plt.cm.gray_r)
    cv2.imshow("Render", color)
    print("done")
    cv2.waitKey(0)
    # plt.show()


if __name__ == "__main__":
    test_render()