import argparse
from robotpose.render import Aligner

def align(dataset, skeleton):

    objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS','MH5_BT_UNIFIED_AXIS']
    names = ['BASE','S','L','U','R','BT']

    align = Aligner(objs, names, dataset, skeleton)
    align.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default="set6", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('skeleton', type=str, default="B", help="The skeleton to use for annotation.")
    args = parser.parse_args()
    align(args.dataset, args.skeleton)