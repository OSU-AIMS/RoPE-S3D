# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley
# Based on "FwdKinematic_MH5L_AllJoints" authored by acbuynak

import json
import os

import numpy as np

from ..paths import Paths as p
from ..urdf import URDFReader


class ForwardKinematics():

    def __init__(self) -> None:
        while not self._getParams(): pass

        self.aa    = np.array(self.params['a'])
        self.alpha = np.array(self.params['alpha'])
        self.dd    = np.array(self.params['d'])

    def _getParams(self):
        u_reader = URDFReader()

        if os.path.isfile(p().DH_PARAMS):
            with open(p().DH_PARAMS,'r') as f:
                config = json.load(f)
        else:
            config = {}

        if u_reader.name in config:
            self.params = config[u_reader.name]
            return True
        else:
            u_reader.guessDHParams()
            return False

    def calc(self, p_in):
        """
        Performs Forward Kinematic Calculation to find the xyz (euler) position of each joint. Rotations NOT output.
        Based on FwdKinematic_MH5L_AllJoints created by acbuynak.
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

        pheta = [0, p_in[0], p_in[1]-1.57079, -p_in[2], p_in[3], p_in[4], p_in[5]]

        ## Forward Kinematics
        # Ref. L13.P5
        T_x = np.zeros((6,4,4))
        T_x[0] = bigMatrix(self.aa[0], self.alpha[0], self.dd[1], pheta[1])

        for i in range(1,6):    # Apply transforms in order, yielding each sucessive frame
            T_x[i] = np.matmul(T_x[i-1], bigMatrix(self.aa[i], self.alpha[i], self.dd[i+1], pheta[i+1]))

        # Extract list of vectors between frames
        vectors = T_x[:, :-1, 3]

        return vectors
