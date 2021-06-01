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
# Refactored by Adam Exley

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
        cosp = np.cos(pheta)
        sinp = np.sin(pheta)
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        T = np.array([[cosp, -sinp, 0, a],
                        [sinp * cosa, cosp * cosa, -sina, -d * sina],
                        [sinp * sina, cosp * sina, cosa, d * cosa],
                        [0, 0, 0, 1]])
        return T

    link  = [0, 1, 2, 3, 4, 5, 6 ]
    aa    = [0, .088, .400, .040, 0, 0, 0]
    alpha = [0, -1.57079, 0, 1.57079, 1.57079, -1.57079, 3.14159]
    dd    = [0, 0, 0, 0, -.405, 0, -.0865]
    pheta = [0, p_in[0], p_in[1]-1.57079, -p_in[2], p_in[3], p_in[4], p_in[5]]


    ## Forward Kinematics
    # Ref. L13.P5
    T_x = np.zeros((6,4,4))
    T_x[0] = bigMatrix(aa[0], alpha[0], dd[1], pheta[1])

    for i in range(1,6):    # Apply transforms in order, yielding each sucessive frame
        T_x[i] = np.matmul(T_x[i-1], bigMatrix(aa[i], alpha[i], dd[i+1], pheta[i+1]))

    # Extract list of vectors between frames
    vectors = T_x[:, :-1, 3]

    return vectors