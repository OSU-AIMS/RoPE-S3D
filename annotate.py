from robotpose.dataset import Dataset
from deepposekit import Annotator
from deepposekit.io import DataGenerator
import os
import argparse

def anno(set,skele):
    ds = Dataset(set,skele)

    print("Annotating on the following dataset:")
    print(ds)

    # Create DeepPoseDataset path if not already made
    ds.makeDeepPoseDS()

    #data_generator = DataGenerator(ds.deepposeds_path, mode="full")


    app = Annotator(datapath=os.path.abspath(ds.deepposeds_path),
                    dataset='images',
                    skeleton=ds.skeleton_path,
                    shuffle_colors=False,
                    text_scale=1)

    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--set', type=str, default="set3", help="The dataset to load to annotate. Can be a partial name.")
    parser.add_argument('--skeleton', '--skele', type=str, default="A", help="The skeleton to use for annotation.")
    args = parser.parse_args()
    anno(args.set, args.skeleton)

"""
+- = rescale image by ±10%
left mouse = move active keypoint to cursor location
WASD = move active keypoint 1px or 10px
space = change WASD mode (swaps between 1px or 10px movements)
JL = next or previous image
<> = jump 10 images forward or backward
I,K or tab, sh/tab = switch active keypoint
R = mark image as unannotated ("reset")
F = mark image as annotated ("finished")
V = mark active keypoint as visible
esc/q = quit
"""