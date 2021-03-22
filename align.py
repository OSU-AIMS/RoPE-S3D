from robotpose.render import Aligner

def align():

    objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS_NEW','MH5_BT_UNIFIED_AXIS']
    names = ['BASE','S','L','U','R','BT']

    align = Aligner(objs, names, 'set6','B')
    align.run()

align()