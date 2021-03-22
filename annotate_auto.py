from robotpose.autoAnnotate import AutomaticKeypointAnnotator
def auto_keypoint():

    objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS_NEW','MH5_BT_UNIFIED_AXIS']
    names = ['BASE','S','L','U','R','BT']

    anno = AutomaticKeypointAnnotator(objs, names, 'set6','B')
    anno.run()



auto_keypoint()