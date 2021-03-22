import json
import numpy as np
import argparse
import os



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path',type=str)
    args = parser.parse_args()

    if os.path.isdir(args.path):
        os.mkdir(os.path.join(args.path,'json_cvt'))
        jsons = [x for x in os.listdir(args.path) if x.endswith('.json')]
        for j in jsons:
            p, xyz = read_old_json(os.path.join(args.path, j))
            write_json(p,xyz.tolist(), os.path.join(os.path.join(args.path,'json_cvt'),j))





def write_json(pose,joint_xyz_points,filename) :
    data = {}
    data['objects'] = []
    data['objects'].append({
    'class': 'bot_mh5',
    'visibility': 1,
    'location': [0,0,0],
    'joints': [
        {
        'name': 'link_1_s',
        'angle': pose[0],
        'position': joint_xyz_points[0],
        },
        {
        'name': 'link_2_l',
        'angle': pose[1],
        'position': joint_xyz_points[1],
        },
        {
        'name': 'link_3_u',
        'angle': pose[2],
        'position': joint_xyz_points[2],
        },
        {
        'name': 'link_4_r',
        'angle': pose[3],
        'position': joint_xyz_points[3],
        },
        {
        'name': 'link_5_b',
        'angle': pose[4],
        'position': joint_xyz_points[4],
        },
        {
        'name': 'link_6_t',
        'angle': pose[5],
        'position': joint_xyz_points[5],
        }],
    })

    with open( str(filename),'w' ) as outfile:
        json.dump(data,outfile, indent=2, sort_keys=True)

    print( "Recorded Pose Data to "+str(filename))





def read_old_json(filename):

    with open(filename, 'r') as f:
        data = json.load(f)

    pose = []
    for i in range(6):
        pose.append(data['objects'][0]['joint_angles'][i]['angle'])

    xyz = fwd(pose)

    return pose, xyz





def fwd(p_in):
  """
  Performs Forward Kinematic Calculation to find the xyz (euler) position of each joint. Rotations NOT output.
  Method: Denavit-Hartenberg parameters used to generate Transformation Matrices. Translation points extracted from the TF matrix.
  :param p_in: List of 6 joint angles (radians)
  :return vectors: List Numpy Array (6x3) where each row is xyz origin of joints
  """
  
  def bigMatrix(a, alpha, d, pheta):
    T = np.array([[np.cos(pheta), -np.sin(pheta), 0, a],
                  [np.sin(pheta) * np.cos(alpha), np.cos(pheta) * np.cos(alpha), -np.sin(alpha), -d * np.sin(alpha)],
                  [np.sin(pheta) * np.sin(alpha), np.cos(pheta) * np.sin(alpha), np.cos(alpha), d * np.cos(alpha)],
                  [0, 0, 0, 1]])
    return T
  
  link  = [0, 1, 2, 3, 4, 5, 6 ]
  aa    = [0, 88, 400, 40, 0, 0, 0]
  alpha = [0, -1.57079, 0, 1.57079, 1.57079, -1.57079, 3.14159]
  dd    = [0, 0, 0, 0, -405, 0, -86.5]
  pheta = [0, p_in[0], p_in[1]-1.57079, -p_in[2], p_in[3], p_in[4], p_in[5]]


  ## Forward Kinematics
  # Ref. L13.P5
  T_01 = bigMatrix(aa[0], alpha[0], dd[1], pheta[1])
  T_12 = bigMatrix(aa[1], alpha[1], dd[2], pheta[2])
  T_23 = bigMatrix(aa[2], alpha[2], dd[3], pheta[3])
  T_34 = bigMatrix(aa[3], alpha[3], dd[4], pheta[4])
  T_45 = bigMatrix(aa[4], alpha[4], dd[5], pheta[5])
  T_56 = bigMatrix(aa[5], alpha[5], dd[6], pheta[6])

  # Create list of Transforms for Frames {i} relative to Base {0}
  T_02 = np.matmul(T_01, T_12)
  T_03 = np.matmul(T_02, T_23)
  T_04 = np.matmul(T_03, T_34)
  T_05 = np.matmul(T_04, T_45)
  T_06 = np.matmul(T_05, T_56)

  # Combine Into List of Transform to allow iteration & easy usage!
  T_combined = [T_01, T_02, T_03, T_04, T_05, T_06]

  # Generate list of vectors between frames
  vectors = np.array(np.transpose(T_combined[0][:-1, 3]))
  vectors = np.vstack((vectors, np.transpose(T_combined[1][:-1, 3])))
  vectors = np.vstack((vectors, np.transpose(T_combined[2][:-1, 3])))
  vectors = np.vstack((vectors, np.transpose(T_combined[3][:-1, 3])))
  vectors = np.vstack((vectors, np.transpose(T_combined[4][:-1, 3])))
  vectors = np.vstack((vectors, np.transpose(T_combined[5][:-1, 3])))
  
  # Switch from mm to meters
  vectors = vectors / 1000

  return vectors


if __name__ == "__main__":
    main()