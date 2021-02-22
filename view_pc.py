import numpy as np
import open3d as o3d

def main():
    cloud = o3d.io.read_triangle_mesh(r'C:\Users\exley\OneDrive - The Ohio State University\CDME\RobotPose\data\ply\2021020600068.ply')
    o3d.visualization.draw_geometries([cloud]) # Visualize the point cloud


if __name__ == "__main__":
    main()

    