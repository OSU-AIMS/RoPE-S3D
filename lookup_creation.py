from robotpose.data import dataset
from robotpose.prediction.predict import LookupCreator
from robotpose import Dataset

ds = Dataset('set10')

lc = LookupCreator(ds.camera_pose[0])
lc.load_config(6,[True,True,True,False,False,False],[20,20,20,0,0,0])
lc.run('test1.h5', False)