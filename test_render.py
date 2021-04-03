import cv2
from robotpose.autoAnnotate import AutomaticKeypointAnnotator
from robotpose.render import Aligner, Renderer
#test_render()
#test_render_with_class()



def test_render():

    r = Renderer('set0', 'B')
    r.setMode('key')

    for frame in range(100):
            
        r.setPosesFromDS(frame)
        color,depth = r.render()
        cv2.imshow("Render", color)
        cv2.waitKey(50)


test_render()