from robotpose.data import dataset
from robotpose.experimental.area import LookupCreator
from robotpose import Dataset


ds = Dataset('set10')

lc = LookupCreator(ds.camera_pose[0])
lc.load_config(4,[True,True,False,False,False,False],[50,50,0,0,0,0])
lc.run('test.h5')