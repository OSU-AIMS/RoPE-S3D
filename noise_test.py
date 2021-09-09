import numpy as np
import cv2

from robotpose import DatasetRenderer
from robotpose.utils import color_array

rend =  DatasetRenderer('set70')

tot = np.zeros((720,1280))



for dilation,thresh in zip([5,10,15,20,25],[.8,.9,.95,.97,.99]):

    a = np.random.normal(0,.25,(720,1280))
    a[a < 0] *= -1
    a = np.clip(a,0,1)
    b = np.copy(a)
    b[b<thresh] = 0

    c = cv2.dilate(b,np.ones((dilation,dilation)))

    #c*=a*2

    tot+=c


connect = 10

tot[tot!=0] = 1

tot = cv2.erode(cv2.dilate(tot,np.ones((connect,connect))),np.ones((connect,connect)))


cv2.imshow("",tot)
cv2.waitKey(0)

tot = tot != 0
tot = tot == 0

c,d = rend.render_at(123)

d *= tot

d *= np.random.normal(1,.01,(720,1280))

cv2.imshow("",color_array(d))
cv2.waitKey(0)