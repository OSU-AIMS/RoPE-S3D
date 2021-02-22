# DeepPoseRobot, an implementation of DeepPoseKit
<p align="center">
<img src="https://github.com/AdamExley/DeepPoseRobot/blob/main/assets/video_overlay_rebound.gif" height="200px">
</p>

This is an adaptation of [DeepPoseKit](deepposekit.org) to predict robot joint angles.

## For more details on DeepPoseKit:

- See [our example notebooks](https://github.com/jgraving/deepposekit/blob/master/examples/)
- Check the [documentation](http://docs.deepposekit.org)
- Read [our paper](https://doi.org/10.7554/eLife.47994)


# Installation

DeepPoseKit requires [Tensorflow](https://github.com/tensorflow/tensorflow) for training and using pose estimation models. [Tensorflow](https://github.com/tensorflow/tensorflow) should be manually installed, along with dependencies such as CUDA and cuDNN, before installing DeepPoseKit:

- [Tensorflow Installation Instructions](https://www.tensorflow.org/install)
- Any Tensorflow version >=1.13.0 should be compatible (including 2.0).
    - Tensorflow-gpu 1.13.1 is currently the only tested version.

## Installing with Anaconda on Windows

To install DeepPoseKit on Windows, you must first manually install `Shapely`, one of the dependencies for the [imgaug package](https://github.com/aleju/imgaug):
```bash
conda install -c conda-forge shapely
```

Install requirements with pip:
```bash
pip install --update --r requirements.txt
```


# License

Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/deepposekit/blob/master/LICENSE) for details.

# References

If you use DeepPoseKit for your research please cite [our open-access paper](http://paper.deepposekit.org):

    @article{graving2019deepposekit,
             title={DeepPoseKit, a software toolkit for fast and robust animal pose estimation using deep learning},
             author={Graving, Jacob M and Chae, Daniel and Naik, Hemal and Li, Liang and Koger, Benjamin and Costelloe, Blair R and Couzin, Iain D},
             journal={eLife},
             volume={8},
             pages={e47994},
             year={2019},
             publisher={eLife Sciences Publications Limited}
             url={https://doi.org/10.7554/eLife.47994},
             }

You can also read [our open-access preprint](http://preprint.deepposekit.org).

If you use the [imgaug package](https://github.com/aleju/imgaug) for data augmentation, please also consider [citing it](https://github.com/aleju/imgaug/blob/master/README.md#citation).

Please also consider citing the relevant references for the pose estimation model(s) used in your research, which can be found in the documentation (i.e., [`StackedDenseNet`](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/StackedDenseNet.html#references), [`StackedHourglass`](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/StackedHourglass.html#references), [`DeepLabCut`](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/DeepLabCut.html#references), [`LEAP`](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/LEAP.html#references)).
