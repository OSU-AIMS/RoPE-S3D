from ..simulation.noise import NoiseMaker
from ..utils import str_to_arr
from ..urdf import URDFReader
from .predict import Predictor
from ..simulation.render import Renderer
import numpy as np
from tqdm import tqdm


class SyntheticPredictor():
    def __init__(self, camera_pose, base_intrin, ds_factor, do_angles, noise):
        self.renderer = Renderer(camera_pose=camera_pose, camera_intrin=base_intrin)
        self.predictor = Predictor(camera_pose, ds_factor,
            do_angles=do_angles, base_intrin=base_intrin,
            color_dict=self.renderer.color_dict)
        self.urdf_reader = URDFReader()
        self.do_angles = do_angles
        self.noise = NoiseMaker()
        self.do_noise = noise


    def run(self):
        pose = self._generatePose()

        self.renderer.setJointAngles(pose)
        color, depth = self.renderer.render()

        # Add noise to depth
        if self.do_noise:
            depth = self.noise.holes(depth,)
        
        predicted = self.predictor.run(color, depth)

        return pose, predicted


    def _generatePose(self):
        selection = np.random.uniform(self.urdf_reader.joint_limits[:,0],self.urdf_reader.joint_limits[:,1])
        selection *= str_to_arr(self.do_angles)
        return selection


    def run_batch(self, number:int, file:str = 'synth_test'):

        if not file.endswith('.npy'):
            file += '.npy'

        # Actual, Predicted
        results = np.zeros((2,number,6))

        for i in tqdm(range(number)):
            results[0,i], results[1,i] = self.run()
            if i % 250 == 0:
                np.save(file,results)

        np.save(file,results)
            