from robotpose import Dataset, SyntheticPredictor
from tqdm import tqdm

angs = 'SLU'
dataset = 'set50'


ds = Dataset(dataset)
synth = SyntheticPredictor(ds.camera_pose[0],'1280_720_color',8,angs)
synth.run_batch(2500)
