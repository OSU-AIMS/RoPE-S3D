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

FILE_TO_CHECK = ""



class JSONCoupling():
    def __init__(self) -> None:
        pass

    def get_pose(self):

        fails = 0

        while True:
            if os.path.isfile(FILE_TO_CHECK):
                try:
                    self.data = json.load(FILE_TO_CHECK)
                    break
                except Exception:
                    fails += 1
                    if fails % 1000 == 0:
                        logging.warning(f"{fails} failures to access JSON coupling file")
                
            time.sleep(0.001)
        
        return self.pose

    def reset(self):

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
                
            time.sleep(0.001)
        


    @property
    def pose(self):
        return [self.data[x] for x in 'SLUBRT']
