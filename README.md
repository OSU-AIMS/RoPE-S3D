# You have just found DeepPoseKit. Kinda.
<p align="center">
<img src="https://github.com/jgraving/jgraving.github.io/blob/master/files/images/Figure1video1.gif" height="128px">
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

Install requirements with pip:
```bash
pip install --update --r requirements.txt
```

## Installing with Anaconda on Windows

To install DeepPoseKit on Windows, you must first manually install `Shapely`, one of the dependencies for the [imgaug package](https://github.com/aleju/imgaug):
```bash
conda install -c conda-forge shapely
```
We also recommend installing DeepPoseKit from within Python rather than using the command line, either from within Jupyter or another IDE, to ensure it is installed in the correct working environment:
```python
import sys
!{sys.executable} -m pip install --update deepposekit
```
# Contributors and Development  
   
DeepPoseKit was developed by [Jake Graving](https://github.com/jgraving) and [Daniel Chae](https://github.com/dchaebae), and is still being actively developed. .

We welcome community involvement and public contributions to the toolkit. If you wish to contribute, please [fork the repository](https://help.github.com/en/articles/fork-a-repo) to make your modifications and [submit a pull request](https://help.github.com/en/articles/creating-a-pull-request-from-a-fork).

If you'd like to get involved with developing DeepPoseKit, get in touch (jgraving@gmail.com) and check out [our development roadmap](https://github.com/jgraving/DeepPoseKit/blob/master/DEVELOPMENT.md) to see future plans for the package.  

# Issues  
 
Please submit bugs or feature requests to the [GitHub issue tracker](https://github.com/jgraving/deepposekit/issues/new). Please limit reported issues to the DeepPoseKit codebase and provide as much detail as you can with a minimal working example if possible.

If you experience problems with [Tensorflow](https://github.com/tensorflow/tensorflow), such as installing CUDA or cuDNN dependencies, then please direct issues to those development teams.

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

If you [use data](https://github.com/jgraving/DeepPoseKit#i-already-have-annotated-data) that was annotated with the DeepLabCut package (http://deeplabcut.org) for your research, be sure to [cite it](https://github.com/AlexEMG/DeepLabCut/blob/master/README.md#references).

Please also consider citing the relevant references for the pose estimation model(s) used in your research, which can be found in the documentation (i.e., [`StackedDenseNet`](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/StackedDenseNet.html#references), [`StackedHourglass`](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/StackedHourglass.html#references), [`DeepLabCut`](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/DeepLabCut.html#references), [`LEAP`](http://jakegraving.com/DeepPoseKit/html/deepposekit/models/LEAP.html#references)).

# News
- **October 2019:** Our paper describing DeepPoseKit is published at eLife! (http://paper.deepposekit.org)
- **September 2019**: 
    - Nature News covers DeepPoseKit: [Deep learning powers a motion-tracking revolution](http://doi.org/10.1038/d41586-019-02942-5)
    - v0.3.0 is released. See [the release notes](https://github.com/jgraving/DeepPoseKit/releases/tag/v0.3.0).
- **April 2019:** The DeepPoseKit preprint is on biorxiv (http://preprint.deepposekit.org)
