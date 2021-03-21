
from robotpose.render import Aligner
#test_render()
#test_render_with_class()





# def test_render():

#     objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS_NEW','MH5_BT_UNIFIED_AXIS']
#     names = ['BASE','S','L','U','R','BT']

#     r = Renderer(objs, name_list=names)
#     r.setMode('key')
#     color_dict = r.getColorDict()
#     anno = KeypointAnnotator(color_dict,'set6','B')
    

#     for frame in range(100):
            
#         r.setPosesFromDS(frame)
#         color,depth = r.render()
#         anno.annotate(color,frame)
#         cv2.imshow("Render", color)
#         cv2.waitKey(100)


def test_align():

    objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS_NEW','MH5_BT_UNIFIED_AXIS']
    names = ['BASE','S','L','U','R','BT']

    align = Aligner(objs, names, 'set6','B')
    align.run()


test_align()