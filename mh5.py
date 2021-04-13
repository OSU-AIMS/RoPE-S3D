#!/usr/bin/env python

##################################################################
##   Forward Kinematics - Join Position Solver                  ##
##                                                              ##
##   Tool Func: "FwdKinematic_MH5L_AllJoints"                   ##
##   * based on Denavit-Hartenberg parameters                   ##
##   * requires Numpy pacakge                                   ##
##                                                              ##
##   Includes sample usage & visualization code in main().      ##
##   Uses matplotlib                                            ##
##                                                              ##
##################################################################

# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: acbuynak

##################################################################


## Imports
import numpy as np
from numpy.core.numeric import cross


## TOOL CODE #####################################################
def FwdKinematic_MH5L_AllJoints(p_in):
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




## SAMPLE USAGE CODE #############################################
def main():
  
  # Joint Values Input (radians)(robot limits not checked)
  p_input = [0, 0, 0, 1, 1, 0]

  
  ########## STATIC Calculation #############
  if False:
    print("\n------- STATIC SIMULATION --------")
    
    ## Calculation
    print("Forward Kinematic Results")
    vectors = FwdKinematic_MH5L_AllJoints(p_input)
    print(np.trunc(vectors))
    
    ## Visualization
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
  
    # Setup our Figure
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('Forward Kinematics Demo: MH5L')
    ax.set(xlim=(-500,500), ylim=(-500,500), zlim=(0,500))
  
    for joint in vectors:
      ax.scatter(joint[0], joint[1], joint[2])
  
    ax.plot(vectors[:,0], vectors[:,1], vectors[:,2])
  
    plt.show()
  
  
  ########## ANIMATED Simulation #############
  if True:
    print("\n------- ANIMATED SIMULATION --------")
    ## Imports
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    from matplotlib.animation import FuncAnimation
  
  
    # Setup our Figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_title('Forward Kinematics Demo: MH5L')
    ax.set(xlim=(0,.75), ylim=(-.75/2, .75/2), zlim=(0, .75))
  
  
    # Generate Data for Animation
    n = 200   # how many frames to include in animation
    p_animate = np.zeros((n,6))
    for i, p in enumerate(p_input):
      p_animate[:,i] = np.linspace(0,p_input[i],n)
  
    # Stack each new KeyFrame into 3rd Dimension of Array. Further back is newer frame.
    animate_vectors = FwdKinematic_MH5L_AllJoints([0,0,0,0,0,0])
    for p in p_animate:
      animate_vectors = np.dstack((animate_vectors, FwdKinematic_MH5L_AllJoints(p)))

    old = np.copy(animate_vectors)
    t = old[5]
    b = old[4]
    u = old[2]

    bt = t - b
    ub = b - u


    cross_p = np.zeros((3,old.shape[2]))
    cross_p_ = np.zeros((3,old.shape[2]))
    for idx in range(old.shape[2]):
        cross_p[...,idx] = np.cross(bt[...,idx],ub[...,idx])
        cross_p_[...,idx] = np.cross(bt[...,idx],cross_p[...,idx])
        cross_p[...,idx] = .1*cross_p[...,idx]/np.linalg.norm(cross_p[...,idx])
        cross_p_[...,idx] = .1*cross_p_[...,idx]/np.linalg.norm(cross_p_[...,idx])



    animate_vectors = np.zeros((old.shape[0]+5,3,old.shape[2]))
    animate_vectors[:6,:,:] = old
    animate_vectors[6,:,:] = b + cross_p
    animate_vectors[7,:,:] = b + cross_p_
    animate_vectors[8,:,:] = b
    animate_vectors[9,:,:] = b + cross_p_
    animate_vectors[10,:,:] = t
    


    # proj_r_b = 
    # for idx in range(pos.shape[0]):
    #     proj_r_b[idx] = (np.dot(r[idx], b[idx])/np.dot(b[idx], b[idx]))*b[idx]

    # z = r - proj_r_b



    print("Resultant Data Array Shape:", animate_vectors.shape)
    print("")
  
    # Setup Animation
    slice = 0  #starting point to setup the figure
    scatter = ax.scatter(animate_vectors[:,0,slice], animate_vectors[:,1,slice], animate_vectors[:,2,slice])
    line = ax.plot(animate_vectors[:, 0, slice], animate_vectors[:, 1, slice], animate_vectors[:, 2, slice])[0]
  
  
    # Animation Controller
    def animate(slice):
      line.set_data(animate_vectors[:, 0, slice], animate_vectors[:, 1, slice])
      line.set_3d_properties(animate_vectors[:, 2, slice])
   
      scatter._offsets3d = (animate_vectors[:,0,slice], animate_vectors[:,1,slice], animate_vectors[:,2,slice])
  
    # Animator!
    animatedFigure = FuncAnimation(
      fig, animate, interval=50, frames=n
    )
  
    plt.show()



if __name__ == '__main__':
    main()

# EOF