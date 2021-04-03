from robotpose.dataset import Dataset
import numpy as np
import cv2
import open3d as o3d
from robotpose.turbo_colormap import color_array

def main():
    cloud = o3d.io.read_point_cloud(r'C:\Users\exley\OneDrive\Documents\GitHub\DeepPoseRobot\data\raw\set5_slu\2021022300005.ply')
    #cloud = o3d.io.read_point_cloud(r'C:\Users\exley\Downloads\tless_models.tar\tless_models\tless_models\obj_01.xyz')
    
    arr = np.asarray(cloud.points)
    #print(arr)
    o3d.visualization.draw_geometries([cloud]) # Visualize the point cloud


def test():
    ds = Dataset('set0','B')
    for idx in range(ds.length):
        cv2.imshow("test",color_array(ds.pointmaps[idx,...,0],.2,3))
        #cv2.imshow("test",np.abs(ds.pointmaps[idx,...,2]-2))
        cv2.waitKey(150)


if __name__ == "__main__":
    #main()
    test()

    