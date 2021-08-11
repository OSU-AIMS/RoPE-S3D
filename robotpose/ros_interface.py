# Software License Agreement (Apache 2.0 License)
#
# Copyright (c) 2021, The Ohio State University
# Center for Design and Manufacturing Excellence (CDME)
# The Artificially Intelligent Manufacturing Systems Lab (AIMS)
# All rights reserved.
#
# Author: Adam Exley

import klampt
import klampt.io.numpy_convert as kionp
import klampt.io.ros as kioros





class Publisher():


    def __init__(self) -> None:
        pass



def callback(data):
    pass

class Subscriber():

    def __init__(self) -> None:
        self.sub = kioros.subscriber("A","Config",callback)
