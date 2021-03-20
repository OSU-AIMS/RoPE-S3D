# DeepPoseRobot, an implementation of DeepPoseKit

To see rendering examples, open [Render_Examples.md](https://github.com/AdamExley/DeepPoseRobot/blob/main/Render_Examples.md)

This is an adaptation of both [DeepPoseKit](deepposekit.org) and [PixelLib](https://github.com/ayoolaolafenwa/PixelLib) to predict robot joint angles.

The robot is isolated from the background using PixelLib and then the keypoint locations of the robot are predicted using a DeepPoseKit model.


Visualization uses the [Turbo Colormap](https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html).

3D Rendering is done via [Pyrender](https://github.com/mmatl/pyrender).

# Automatic Annotation

## Alignment

Before running any automatic annotation, first align the dataset with the render using the Aligner class:

```python
from robotpose.render import Aligner
align = Aligner('example_dataset_name','example_skeleton_name')
align.run()
```

## Keypoints

To run automatic keypoint annotation, first align the dataset.

Then, use the AutomaticKeypointAnnotator class to annotate:

```python
from robotpose.autoAnnotate import AutomaticKeypointAnnotator
# Mesh information
objs = ['MH5_BASE', 'MH5_S_AXIS','MH5_L_AXIS','MH5_U_AXIS','MH5_R_AXIS_NEW','MH5_BT_UNIFIED_AXIS']
names = ['BASE','S','L','U','R','BT']
anno = AutomaticKeypointAnnotator(objs, names, 'example_dataset_name','example_skeleton_name')
anno.run()
```




# Installation

This requires [Tensorflow](https://github.com/tensorflow/tensorflow) for both segmentation and pose estimation. [Tensorflow](https://github.com/tensorflow/tensorflow) should be manually installed, along with CUDA and cuDNN as follows:

- [Tensorflow Installation Instructions](https://www.tensorflow.org/install)
- Any Tensorflow version >=2.0.0 should be compatible.
    - Tensorflow-gpu 2.0.0 is currently the only tested version.

## Installing with Anaconda on Windows

To use DeepPoseKit on Windows, you must first manually install `Shapely`, one of the dependencies for the [imgaug package](https://github.com/aleju/imgaug):
```bash
conda install -c conda-forge shapely
```

Install requirements with pip:
```bash
pip install --upgrade --r requirements.txt
```
Sometimes Pixellib will not work after all installations have been completed. To fix this error, upgrade and downgrade Tensorflow.
```bash
pip install --upgrade tensorflow-gpu
pip install --upgrade tensorflow-gpu==2.0.0
```


# License

Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/deepposekit/blob/master/LICENSE) for details.
