##################################################################
##   Forward Kinematics - Join Position Solver                  ##
##                                                              ##
##   Tool Func: "FwdKinematic_MH5L_AllJoints"                   ##
##   * based on Denavit-Hartenberg parameters                   ##
##   * requires Numpy pacakge                                   ##
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

# EOF