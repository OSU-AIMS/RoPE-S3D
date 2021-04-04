import argparse
from tokenize import endpats
from robotpose.render import Aligner

def align(dataset, skeleton, start, end):

    align = Aligner(dataset, skeleton, start, end)
    align.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, default="set6", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('skeleton', type=str, default="B", help="The skeleton to use for annotation.")
    parser.add_argument('--start', type=int, default=None, help="Starting index.")
    parser.add_argument('--end', type=int, default=None, help="Ending index.")
    args = parser.parse_args()
    align(args.dataset, args.skeleton, args.start, args.end)