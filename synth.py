from robotpose import Dataset, SyntheticPredictor
from tqdm import tqdm

angs = 'SLU'
dataset = 'set30'


ds = Dataset(dataset)
print(ds.camera_pose[0])
synth = SyntheticPredictor(ds.camera_pose[0],'640_480_color',2,angs)
synth.run_batch(5000)
