import robotpose
from robotpose.simulation.lookup import RobotLookupManager
from robotpose import Dataset
from robotpose.utils import get_extremes
from robotpose.crop import Crop
from robotpose.projection import Intrinsics

ds = Dataset('set21')
# lm = RobotLookupManager()
# ang, depth = lm.get(str(ds.intrinsics),ds.camera_pose[0],3,'SL')
# print(get_extremes(depth[0] != 0))
i = Intrinsics(str(ds.intrinsics))
# i.downscale(4)
print(Intrinsics(i).size)
c = Crop(ds.camera_pose[0],str(i))

print(c[1])