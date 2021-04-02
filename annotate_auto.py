import argparse
from robotpose.render import Renderer
from robotpose.autoAnnotate import AutomaticKeypointAnnotator, AutomaticSegmentationAnnotator

def label(dataset, skeleton, preview):

    objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS','MH5_BT_UNIFIED_AXIS']
    names = ['BASE','S','L','U','R','BT']

    rend = Renderer(objs, dataset, skeleton, names)

    key = AutomaticKeypointAnnotator(objs, names, dataset, skeleton, renderer = rend, preview = preview)
    key.run()

    del key

    seg = AutomaticSegmentationAnnotator(objs, names, dataset, skeleton, renderer = rend, preview = preview)
    seg.run()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default="set6", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('skeleton', type=str, default="B", help="The skeleton to use for annotation.")
    parser.add_argument('--no_preview', action="store_true", help="Disables preview.")
    args = parser.parse_args()
    label(args.dataset, args.skeleton, not args.no_preview)