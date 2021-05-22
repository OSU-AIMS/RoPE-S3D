from OpenGL.raw.GL.VERSION.GL_3_0 import GL_FRAMEBUFFER_ATTACHMENT_ALPHA_SIZE
from tensorflow.python.ops.gen_math_ops import TruncateMod
from robotpose.prediction.predict import LookupCreator
from robotpose import Dataset

ds = Dataset('set10')

lc = LookupCreator(ds.camera_pose[0],8)
lc.load_config(3,[True,True,False,False,False,False],[100,100,0,0,0,0])
lc.run('SL100.h5', False)