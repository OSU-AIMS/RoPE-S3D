from robotpose.dataset import Dataset
import matplotlib.pyplot as plt
from deepposekit.io import DataGenerator, initialize_dataset
import os


skeleton_path = r"models/skeletons/A.csv"

ds = Dataset('set3')

if not os.path.isfile(ds.deepposeds_path):
    initialize_dataset(
        images=ds.rm_img,
        datapath=ds.deepposeds_path,
        skeleton=skeleton_path,
        overwrite=False # This overwrites the existing datapath
    )

data_generator = DataGenerator(ds.deepposeds_path, mode="full")


from deepposekit import Annotator
app = Annotator(datapath=os.path.abspath(ds.deepposeds_path),
                dataset='images',
                skeleton=skeleton_path,
                shuffle_colors=False,
                text_scale=1)

app.run()


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