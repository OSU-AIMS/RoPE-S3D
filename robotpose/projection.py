import os
import json
import numpy as np
import cv2
import pyrealsense2 as rs
from tqdm import tqdm
import open3d as o3d


FLT_EPSILON = 1





def makeIntrinsics(preset = '1280_720_color'):
    """
    Make Realsense Intrinsics from presets
    """

    valid = ['1280_720_color', '1280_720_depth','640_480_color','640_480_depth']
    if preset not in valid:
        raise ValueError(f"Res must be one of: {valid}")

    if preset == '1280_720_color':
        return intrin((1280,720), (638.391,361.493), (905.23, 904.858), rs.distortion.inverse_brown_conrady, [0,0,0,0,0])
    elif preset == '1280_720_depth':
        return intrin((1280,720), (639.459,359.856), (635.956, 635.956),rs.distortion.brown_conrady, [0,0,0,0,0])
    elif preset == '640_480_color':
        return intrin((640,480), (320.503,237.288), (611.528,611.528),rs.distortion.brown_conrady, [0,0,0,0,0])
    elif preset == '640_480_depth':
        return intrin((640,480), (321.635,241.618), (385.134,385.134),rs.distortion.brown_conrady, [0,0,0,0,0])




def intrin(resolution, pp, f, model, coeffs):
    """
    Makes psuedo-intrinsics for the realsense camera used.
    """
    a = rs.intrinsics()
    a.width = max(resolution)
    a.height = min(resolution)
    a.ppx = pp[0]
    a.ppy = pp[1]
    a.fx = f[0]
    a.fy = f[1]
    a.coeffs = coeffs
    a.model = model
    return a





def proj_point_to_pixel(intrin, points, correct_distortion = False):
    """
    Python copy of the C++ realsense sdk function
    Can take arrays as inputs to speed up calculations
    Expects n x 3 array of points to project
    """
    x = points[:,0] / points[:,2]
    y = points[:,1] / points[:,2]

    if correct_distortion:
        if intrin.model == rs.distortion.inverse_brown_conrady or intrin.model == rs.distortion.modified_brown_conrady:

            r_two = np.square(x) + np.square(y)

            f = 1 + intrin.coeffs[0] * r_two + intrin.coeffs[0] * np.square(r_two) + intrin.coeffs[4] * np.power(r_two,3)

            x *= f
            y *= f

            dx = x + 2*intrin.coeffs[2]*x*y + intrin.coeffs[3]*(r_two + 2*np.square(x))
            dy = y + 2*intrin.coeffs[3]*x*y + intrin.coeffs[2]*(r_two + 2*np.square(y))

            x = dx
            y = dy

        elif intrin.model == rs.distortion.brown_conrady:

            r_two = np.square(x) + np.square(y)

            f = 1 + intrin.coeffs[0] * r_two + intrin.coeffs[1] * np.square(r_two) + intrin.coeffs[4] * np.power(r_two,3)

            xf = x*f
            yf = y*f

            dx = xf + 2 * intrin.coeffs[2] * x*y + intrin.coeffs[3] * (r_two + 2 * np.square(x))
            dy = yf + 2 * intrin.coeffs[3] * x*y + intrin.coeffs[2] * (r_two + 2 * np.square(y))

            x = dx
            y = dy

        elif intrin.model == rs.distortion.ftheta:
            r = np.sqrt(np.square(x) + np.square(y))

            if r < FLT_EPSILON:
                r = FLT_EPSILON

            rd = (1.0 / intrin.coeffs[0] * np.arctan(2 * r* np.tan(intrin.coeffs[0] / 2.0)))

            x *= rd / r
            y *= rd / r

        elif intrin.model == rs.distortion.kannala_brandt4:

            r = np.sqrt(np.square(x) + np.square(y))

            if (r < FLT_EPSILON):
                r = FLT_EPSILON

            theta = np.arctan(r)

            theta_two = np.square(theta)

            series = 1 + theta_two*(intrin.coeffs[0] + theta_two*(intrin.coeffs[1] + theta_two*(intrin.coeffs[2] + theta_two*intrin.coeffs[3])))

            rd = theta*series
            x *= rd / r
            y *= rd / r

    pixel = np.zeros((points.shape[0],2))
    pixel[:,0] = x * intrin.fx + intrin.ppx
    pixel[:,1] = y * intrin.fy + intrin.ppy

    return pixel








def test_render():
    import numpy as np
    import trimesh
    import pyrender
    import matplotlib.pyplot as plt

    path = r'C:\Users\exley\OneDrive - The Ohio State University\CDME\RobotPose\robot_models\ply\textured.obj'
    path = r'C:\Users\exley\OneDrive - The Ohio State University\CDME\RobotPose\robot_models\ply\MH5_R_AXIS.ply'
    #path = r'C:\Users\exley\OneDrive - The Ohio State University\CDME\RobotPose\robot_models\ply\banana.obj'
    #path = r'C:\Users\exley\OneDrive - The Ohio State University\CDME\RobotPose\robot_models\ply\a.stl'

    with open(path, 'rb') as f:
        boxv_trimesh = trimesh.exchange.ply.load_ply(f)

    print(boxv_trimesh.keys())

    mesh = pyrender.Mesh.from_points(boxv_trimesh['vertices'])

    #boxv_trimesh = trimesh.load(path)
    #boxv_trimesh = trimesh.creation.box(extents=0.1*np.ones(3))
    # boxv_vertex_colors = np.random.uniform(size=(boxv_trimesh.vertices.shape))
    # boxv_trimesh.visual.vertex_colors = boxv_vertex_colors
    # boxv_mesh = pyrender.Mesh.from_trimesh(boxv_trimesh, smooth=False)

    # tm = trimesh.load(path)
    # #fuze_trimesh = trimesh.load('examples/models/fuze.obj')
    # #tm.visual.vertex_colors = np.random.uniform(size=tm.vertices.shape)
    # tm.visual.face_colors = np.random.uniform(size=tm.faces.shape)
    # pts = tm.vertices.copy()
    # print(pts.shape)
    # #mesh = pyrender.Mesh.from_trimesh(tm,smooth=False)
    # mesh = pyrender.Mesh.from_points(tm.vertices.copy())
    # print(mesh.primitives)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    #s = np.sqrt(2)/2
    s = np.sqrt(2)/2
    camera_pose = np.array([
        [0.0, -s,   s,   2],
        [1.0,  0.0, 0.0, 0.0],
        [0.0,  s,   s,   1],
        [0.0,  0.0, 0.0, 1.0],
     ])
    scene.add(camera, pose=camera_pose)
    light = pyrender.SpotLight(color=np.ones(3), intensity=50.0,
                                innerConeAngle=np.pi/16.0,
                                outerConeAngle=np.pi/6.0)
    scene.add(light, pose=camera_pose)
    r = pyrender.OffscreenRenderer(800, 800)
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




