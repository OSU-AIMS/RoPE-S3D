import numpy as np
import open3d as o3d

def main():
    cloud = o3d.io.read_point_cloud("C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\data\\set_sl\\2021020600068.ply") # Read the point cloud
    cloud = o3d.io.read_triangle_mesh("C:\\Users\\exley\\Desktop\\CDME\\RobotPose\\robot_models\\ply\\high_res\\MH5_B0_AXIS.ply")
    cloud.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([cloud]) # Visualize the point cloud
    print(cloud.has_colors())


if __name__ == "__main__":
    main()

    