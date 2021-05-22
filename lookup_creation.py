from OpenGL.raw.GL.VERSION.GL_3_0 import GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE
from tensorflow.python.ops.gen_math_ops import TruncateMod
from robotpose.prediction.predict import LookupCreator
from robotpose import Dataset

ds = Dataset('set10')

lc = LookupCreator(ds.camera_pose[0],8)
lc.load_config(6,[True,True,True,False,False,False],[50,50,30,0,0,0])
lc.run('SLU50.h5', False)