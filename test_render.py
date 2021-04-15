import cv2
import numpy as np
from robotpose.render import DatasetRenderer



def save_video(path, img_arr):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path,fourcc, 30, (img_arr.shape[2],img_arr.shape[1]))
    for img in img_arr:
        out.write(img)
    out.release()

def test_render():

    r = DatasetRenderer('set10', 'B')
    r.setMode('key')

    color_frames =[]

    for frame in range(r.ds.length):
            
        r.setPosesFromDS(frame)
        color,depth = r.render(True)
        color_frames.append(color)
        cv2.imshow("Render", color)
        cv2.waitKey(50)

    save_video('output/test_render.avi',np.array(color_frames))


test_render()