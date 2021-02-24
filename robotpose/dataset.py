import os
import cv2
import robotpose.utils as utils
import numpy as np
from tqdm import tqdm
import json
import pyrealsense2 as rs
import open3d as o3d
import pickle


def build(data_path, dest_path):

    # Make dataset folder if it does not already exist
    if not os.path.isdir(dest_path):
        os.mkdir(dest_path)

    # Build lists of files
    jsons = [x for x in os.listdir(data_path) if x.endswith('.json')]
    plys = [x for x in os.listdir(data_path) if x.endswith('.ply')]
    orig_img = [x for x in os.listdir(data_path) if x.endswith('og.png')]
    rm_img = [x for x in os.listdir(data_path) if x.endswith('rm.png')]


    # Determine if orig/rm images are being used
    use_orig = use_rm = True
    if len(orig_img) == 0:
        use_orig = False

    if len(rm_img) == 0:
        use_rm = False

    # Check that 2D images were provided
    assert use_orig or use_rm, "No images provided."

    # Make sure number of rm and orig images is the same, if applicable
    if use_rm and use_orig:
        assert len(orig_img) == len(rm_img), "Unequal number of removed and original images."

    # Make sure overall dataset length is the same for each file type
    length = max(len(rm_img), len(orig_img))
    assert len(jsons) == len(plys) == length, "Unequal number of images, jsons, or plys"


    """
    Parse Images
    """
    # Read in orig images if provided
    if use_orig:

        # Get image dims
        img = cv2.imread(os.path.join(data_path,orig_img[0]))
        img_height = img.shape[0]
        img_width = img.shape[1]

        # Create image array
        orig_img_arr = np.zeros((length, img_height, img_width, 3), dtype=np.uint8)

        # Get paths for each image
        orig_img_path = [os.path.join(data_path, x) for x in orig_img]

        # Store images in array
        for idx, path in tqdm(zip(range(length), orig_img_path),desc="Parsing Orig 2D Images"):
            orig_img_arr[idx] = cv2.imread(path)

        # Save array
        np.save(os.path.join(dest_path, 'og_img.npy'), orig_img_arr)


    # Read in rm images if provided
    if use_rm:

        # Get image dims
        img = cv2.imread(os.path.join(data_path,rm_img[0]))
        img_height = img.shape[0]
        img_width = img.shape[1]

        # Create image array
        rm_img_arr = np.zeros((length, img_height, img_width, 3), dtype=np.uint8)

        # Get paths for each image
        rm_img_path = [os.path.join(data_path, x) for x in rm_img]

        # Store images in array
        for idx, path in tqdm(zip(range(length), rm_img_path),desc="Parsing Rm 2D Images"):
            rm_img_arr[idx] = cv2.imread(path)

        # Save array
        np.save(os.path.join(dest_path, 'rm_img.npy'), rm_img_arr)


    """
    Parse JSONs
    """
    json_path = [os.path.join(data_path, x) for x in jsons]
    json_arr = np.zeros((length, 6), dtype=float)

    for idx, path in tqdm(zip(range(length), json_path), desc="Parsing JSON Joint Angles"):
        # Open file
        with open(path, 'r') as f:
            d = json.load(f)
        d = d['objects'][0]['joint_angles']

        # Put data in array
        for sub_idx in range(6):
            json_arr[idx,sub_idx] = d[sub_idx]['angle']

    # Save JSON data as npy
    np.save(os.path.join(dest_path, 'ang.npy'), json_arr)


    """
    Parse PLYs as 3D points

    As the number of verticies in each frame varies, these cannot be saved as a .npy file
    Instead, they are saved as a list of dictionaries in each file, and must be processed upon loading
    """
    intrin = utils.makeIntrinsics()
    ply_path = [os.path.join(data_path, x) for x in plys]

    ply_arr = []

    for path in tqdm(ply_path, desc="Parsing PLY data"):
        # Read file
        cloud = o3d.io.read_point_cloud(path)
        points = np.asarray(cloud.points)
        # Invert x Coords
        points[:,0] = points[:,0] * -1

        data = np.zeros((points.shape[0], 5))
        data[:,2:5] = points

        # Get pixel location of point
        for row in range(points.shape[0]):
            x, y = rs.rs2_project_point_to_pixel(intrin, data[row,2:5])
            data[row,0:2] = [x,y]

        ply_arr.append(data)

    # Save as pickle
    with open(os.path.join(dest_path,'ply.pyc'),'wb') as file:
        pickle.dump(ply_arr,file)


    """
    Write dataset info file
    """
    # Make json info file
    info = {
        "name": os.path.basename(os.path.normpath(dest_path)),
        "frames": length,
        "use_orig": use_orig,
        "use_rm": use_rm
    }

    with open(os.path.join(dest_path,'ds.json'),'w') as file:
        json.dump(info, file)
