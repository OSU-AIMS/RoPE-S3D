# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import json
import os
import time
import logging
import numpy as np

FILE_TO_CHECK = r"\\marvin\ROPE\joint_states.json"



class JSONCoupling():
    def __init__(self) -> None:
        pass

    def get_pose(self, timeout = None):

        start = time.time()

        fails = 0

        while True:
            if os.path.isfile(FILE_TO_CHECK):
                try:
                    with open(FILE_TO_CHECK,'r') as f:
                        self.data = json.load(f)
                    break
                except Exception:
                    fails += 1
                    if fails % 1000 == 0:
                        logging.warning(f"{fails} failures to access JSON coupling file")

            if timeout is not None:
                if time.time() - start > timeout:
                    return None
                
            time.sleep(0.0001)

        return np.array(self.data['position'])

    def reset(self, timeout = None):

        start = time.time

        fails = 0

        while True:
            if os.path.isfile(FILE_TO_CHECK):
                try:
                    os.remove(FILE_TO_CHECK)
                    break
                except Exception:
                    fails += 1
                    if fails % 1000 == 0:
                        logging.warning(f"{fails} failures to delete JSON coupling file")

            if timeout is not None:
                if time.time() - start > timeout:
                    break
                
            time.sleep(0.0001)
        


    @property
    def pose(self):
        return [self.data[x] for x in 'SLUBRT']
