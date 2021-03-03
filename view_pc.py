import numpy as np
import open3d as o3d

def main():
    cloud = o3d.io.read_point_cloud(r'C:\Users\exley\OneDrive\Documents\GitHub\DeepPoseRobot\data\raw\set5_slu\2021022300005.ply')
    #cloud = o3d.io.read_point_cloud(r'C:\Users\exley\Downloads\tless_models.tar\tless_models\tless_models\obj_01.xyz')
    
    arr = np.asarray(cloud.points)
    #print(arr)
    o3d.visualization.draw_geometries([cloud]) # Visualize the point cloud


if __name__ == "__main__":
    main()

    