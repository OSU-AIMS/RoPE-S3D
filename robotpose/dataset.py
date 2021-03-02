import os
import cv2
import robotpose.utils as utils
import numpy as np
from tqdm import tqdm
import json
import pyrealsense2 as rs
import open3d as o3d
import pickle
from robotpose import paths as p


def build(data_path, dest_path = None):

    if dest_path is None:
        name = os.path.basename(os.path.normpath(data_path))
        dest_path = os.path.join(p.datasets, name)

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

        # Save as a video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(dest_path,"og_vid.avi"),fourcc, 15, (img_width,img_height))
        for idx in range(length):
            out.write(orig_img_arr[idx])
        out.release()


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

        # Save as a video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(os.path.join(dest_path,"rm_vid.avi"),fourcc, 15, (img_width,img_height))
        for idx in range(length):
            out.write(rm_img_arr[idx])
        out.release()


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


class Dataset():
    def __init__(self, name, skeleton=None, no_data = False, primary = "rm"):
        # Search for dataset with correct name
        datasets = [ f.path for f in os.scandir(p.datasets) if f.is_dir() ]
        names = [ os.path.basename(os.path.normpath(x)) for x in datasets ]
        ds_found = False
        for ds, nm in zip(datasets, names):
            if name in nm:
                if self.validate(ds):
                    self.path = ds
                    self.name = nm
                    ds_found = True
                    break
                else:
                    print("\nDataset Incomplete.")
                    print(f"Recompiling:\n")
                    self.build(os.path.join(os.path.join(p.datasets,'raw'),nm))
                    self.path = ds
                    self.name = nm
                    ds_found = True
                    break


        # If no dataset was found, try to find one to build
        if not ds_found:
            datasets = [ f.path for f in os.scandir(os.path.join(p.datasets,'raw')) if f.is_dir() ]
            names = [ os.path.basename(os.path.normpath(x)) for x in datasets ]
            for ds, nm in zip(datasets, names):
                if name in nm:
                    print("\nNo matching compiled dataset found.")
                    print(f"Compiling from {ds}:\n")
                    self.build(ds)
                    self.path = os.path.join(p.datasets, nm)
                    self.name = nm
                    ds_found = True
                    break
        
        # Make sure a dataset was found
        assert ds_found, f"No matching dataset found for '{name}'"

        # Load dataset
        self.load(skeleton)

        # Set paths
        self.rm_vid_path = os.path.join(self.path, 'rm_vid.avi')
        self.og_vid_path = os.path.join(self.path, 'og_vid.avi')

        # Set resolution
        if self.use_rm:
            self.resolution = self.rm_img.shape[1:3]
        if self.use_og:
            self.resolution = self.og_img.shape[1:3]

        # Set primary image and video types
        if self.use_rm and not self.use_og:
            primary = "rm"
        if self.use_og and not self.use_rm:
            primary = "og"

        if primary == "og":
            self.img = self.og_img
            self.vid = self.og_vid
            self.vid_path = self.og_vid_path
        else:
            self.img = self.rm_img
            self.vid = self.rm_vid
            self.vid_path = self.rm_vid_path
            if primary != "rm":
                print("Invalid primary media type selected.\nUsing rm.")



        # If specified, remove all data from object to save space (only obtain paths)
        if no_data:
            if self.use_og:
                del self.og_img, self.og_vid
            if self.use_rm:
                del self.rm_img, self.rm_vid
            del self.angles, self.ply


    def load(self, skeleton=None):
        # Read into JSON to get dataset settings
        with open(os.path.join(self.path, 'ds.json'), 'r') as f:
            d = json.load(f)

        self.length = d['frames']
        self.use_og = d['use_orig']
        self.use_rm = d['use_rm']

        # Read in og images
        if self.use_og:
            self.og_img = np.load(os.path.join(self.path, 'og_img.npy'))
            self.og_vid = cv2.VideoCapture(os.path.join(self.path, 'og_vid.avi'))

        # Read in rm images
        if self.use_rm:
            self.rm_img = np.load(os.path.join(self.path, 'rm_img.npy'))
            self.rm_vid = cv2.VideoCapture(os.path.join(self.path, 'rm_vid.avi'))

        # Read angles
        self.angles = np.load(os.path.join(self.path, 'ang.npy'))

        # Read in point data
        with open(os.path.join(self.path, 'ply.pyc'), 'rb') as f:
            self.ply = pickle.load(f)
        # Make sure points are as numpy arrays
        for idx in range(self.length):
            self.ply[idx] = np.asarray(self.ply[idx])

        # Set deeppose dataset path
        self.deepposeds_path = os.path.join(self.path,'deeppose.h5')

        # If a skeleton is set, change paths accordingly
        if skeleton is not None:
            self.setSkeleton(skeleton)


    def validate(self, path):
        ang = os.path.isfile(os.path.join(path,'ang.npy'))
        ds = os.path.isfile(os.path.join(path,'ds.json'))
        ply = os.path.isfile(os.path.join(path,'ply.pyc'))
        rm_img = os.path.isfile(os.path.join(path,'rm_img.npy'))
        og_img = os.path.isfile(os.path.join(path,'og_img.npy'))
        rm_vid = os.path.isfile(os.path.join(path,'rm_vid.avi'))
        og_vid = os.path.isfile(os.path.join(path,'og_vid.avi'))

        return ang and ds and ply and ((rm_img and rm_vid) or (og_img and og_vid))


    def build(self,data_path):
        build(data_path)

    def setSkeleton(self,skeleton_name):
        for file in [x for x in os.listdir(p.skeletons) if x.endswith('.csv')]:
            if skeleton_name in os.path.splitext(file)[0]:
                self.skeleton = os.path.splitext(file)[0]
                self.skeleton_path = os.path.join(p.skeletons, file)
                self.deepposeds_path = self.deepposeds_path.replace('.h5','_'+os.path.splitext(file)[0]+'.h5')


    def __len__(self):
        if self.length is None:
            return 0
        else:
            return self.length  

    def __repr__(self):
        return f"RobotPose dataset of {self.length} frames. Using skeleton {self.skeleton}"

    def og(self):
        return self.use_og

    def rm(self):
        return self.use_rm
