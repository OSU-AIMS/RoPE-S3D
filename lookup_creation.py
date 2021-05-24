from robotpose import Dataset, LookupCreator

ds = Dataset('set10')

lc = LookupCreator(ds.camera_pose[0],8)
lc.load_config(3,[True,True,False,False,False,False],[100,100,0,0,0,0])
lc.run('aaaaa.h5', True)