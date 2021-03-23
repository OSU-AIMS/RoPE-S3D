import cv2
from robotpose.autoAnnotate import AutomaticKeypointAnnotator
from robotpose.render import Aligner, Renderer
#test_render()
#test_render_with_class()



def test_render():

    objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS','MH5_BT_UNIFIED_AXIS']
    names = ['BASE','S','L','U','R','BT']

    r = Renderer(objs, 'set7', 'B',name_list=names)
    r.setMode('seg')

    

    for frame in range(100):
            
        r.setPosesFromDS(frame)
        color,depth = r.render()
        cv2.imshow("Render", color)
        cv2.waitKey(50)


test_render()