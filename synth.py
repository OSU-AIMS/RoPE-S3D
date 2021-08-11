from robotpose import Dataset, SyntheticPredictor
from tqdm import tqdm

angs = 'SLU'
dataset = 'set21'


ds = Dataset(dataset)
print(ds.camera_pose[0])
synth = SyntheticPredictor(ds.camera_pose[0],'1280_720_color',4,angs)
synth.run_batch(5000)
