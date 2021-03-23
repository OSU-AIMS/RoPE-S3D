import argparse
from robotpose.autoAnnotate import AutomaticKeypointAnnotator, AutomaticSegmentationAnnotator

def label(dataset, skeleton):

    objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS','MH5_BT_UNIFIED_AXIS']
    names = ['BASE','S','L','U','R','BT']

    key = AutomaticKeypointAnnotator(objs, names, dataset, skeleton)
    key.run()

    del key

    seg = AutomaticSegmentationAnnotator(objs, names, dataset, skeleton)
    seg.run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default="set6", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('skeleton', type=str, default="B", help="The skeleton to use for annotation.")
    args = parser.parse_args()
    label(args.dataset, args.skeleton)