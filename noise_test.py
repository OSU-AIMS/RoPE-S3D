import numpy as np
import cv2

from robotpose import DatasetRenderer
from robotpose.utils import color_array
from robotpose.simulation.noise import NoiseMaker

rend =  DatasetRenderer('set70')
n = NoiseMaker()


for idx in range(rend.ds.length):
    c,d = rend.render_at(idx)
    d = n.holes(d,25)
    cv2.imshow("",color_array(d))
    cv2.waitKey(0)