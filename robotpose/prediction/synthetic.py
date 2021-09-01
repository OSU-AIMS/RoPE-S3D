from os import truncate
from robotpose import prediction
from robotpose.utils import str_to_arr
from robotpose.urdf import URDFReader
from robotpose.projection import Intrinsics
from .predict import Predictor
from ..simulation.render import Renderer
import numpy as np
from tqdm import tqdm


class SyntheticPredictor():
    def __init__(self, camera_pose, base_intrin, ds_factor, do_angles):
        self.renderer = Renderer(camera_pose=camera_pose, camera_intrin=base_intrin)
        self.predictor = Predictor(camera_pose, ds_factor,
            do_angles=do_angles, base_intrin=base_intrin,
            color_dict=self.renderer.color_dict)
        self.urdf_reader = URDFReader()
        self.do_angles = do_angles


    def run(self):
        pose = self._generatePose()

        self.renderer.setJointAngles(pose)
        color, depth = self.renderer.render()

        # Add gaussian noise
        # depth[depth!=0] += np.random.normal(loc=.3,scale = 1,size = depth[depth!=0].shape)

        predicted = self.predictor.run(color, depth)

        return pose, predicted


    def _generatePose(self):
        selection = np.random.uniform(self.urdf_reader.joint_limits[:,0],self.urdf_reader.joint_limits[:,1])
        selection *= str_to_arr(self.do_angles)
        return selection


    def run_batch(self, number):

        # Actual, Predicted
        results = np.zeros((2,number,6))

        for i in tqdm(range(number)):
            results[0,i], results[1,i] = self.run()
            if i % 250 == 0:
                np.save('synth_test.npy',results)

        np.save('synth_test.npy',results)
            