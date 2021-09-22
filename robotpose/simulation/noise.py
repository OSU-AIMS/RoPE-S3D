from typing import List
import numpy as np
import cv2



class NoiseMaker():

    def holes(self, arr, max_size = 25, std = 0.22, thresh_factor = 1, connection_factor = 20):
        shape = arr.shape

        holes = np.zeros(shape)

        for dilation in np.arange(3,max_size,3):
            thresh = -thresh_factor / dilation + 1
            noise = np.random.normal(0,std,shape)
            noise[noise < 0] *= -1
            noise = np.clip(noise,0,1)
            noise[noise<thresh] = 0
            noise = cv2.dilate(noise,np.ones((dilation,dilation)))

            holes += noise

        holes[holes!=0] = 1

        holes = cv2.erode(cv2.dilate(holes,np.ones((connection_factor,connection_factor))),np.ones((connection_factor,connection_factor)))

        holes = holes != 0
        holes = holes == 0

        return arr * holes.astype(float)


