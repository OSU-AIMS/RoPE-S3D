from robotpose.dataset import Dataset
import numpy as np
import cv2
import open3d as o3d
from robotpose.turbo_colormap import color_array
from robotpose.projection import fill_hole

import matplotlib.pyplot as plt

def main():
    cloud = o3d.io.read_point_cloud(r'C:\Users\exley\OneDrive\Documents\GitHub\DeepPoseRobot\data\raw\set5_slu\2021022300005.ply')
    #cloud = o3d.io.read_point_cloud(r'C:\Users\exley\Downloads\tless_models.tar\tless_models\tless_models\obj_01.xyz')
    
    arr = np.asarray(cloud.points)
    #print(arr)
    o3d.visualization.draw_geometries([cloud]) # Visualize the point cloud


def test():

    r = 300
    c = 400
    errs = []

    

    ds = Dataset('set10','B')
    map = np.copy(ds.pointmaps[...,2])
    for idx in range(ds.length):
        colored = color_array(map[idx])
        
        #print(f"{ds.pointmaps[idx,r,c]} , {fill_hole(ds.pointmaps[idx],r,c,50)}")

        # if np.any(ds.pointmaps[idx,r,c]):
        #     err = (fill_hole(ds.pointmaps[idx],r,c,50) - ds.pointmaps[idx,r,c]) / ds.pointmaps[idx,r,c]
        #     errs.append(err)

        #cv2.circle(colored,(c,r),4,(255,255,255))
        #cv2.imshow("test",np.abs(ds.pointmaps[idx,...,2]-2))
        cv2.imshow("test",colored)
        cv2.waitKey(1)
    # errs = np.array(errs)
    # print(np.mean(errs,0))

    # plt.plot(errs[:,0])
    # plt.plot(errs[:,1])
    # plt.plot(errs[:,2])
    # plt.show()



if __name__ == "__main__":
    #main()
    test()

    