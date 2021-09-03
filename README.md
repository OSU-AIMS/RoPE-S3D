# RoPE-S3D

**Ro**botic **P**ose **E**stimation using **S**egmented **3**-**D**imensional images

<img src="assets/demo.gif" align="right"/>

**For full documentation of usage/setup please follow the steps outlined in [GUIDE.md](GUIDE.md).**

**For prediction-specific documentation please see [PREDICTION.md](PREDICTION.md).**

This tool uses Artifical Intelligence (AI), specifically image segmentation algorithms provided by [PixelLib](https://github.com/ayoolaolafenwa/PixelLib), to predict robot joint angles.

This is acomplished by segmenting the robot and each of its joints from an RGBD (3-D) image. *With a known camera pose realtive to the robot*, these segmented sections of the image can be compared to a 3-D rendering of the robot in any possible pose.

An error function can be defined, quantitatively comparing these images based on visual and depth-based similarity. For any given input, this characterizes loss as a function of the rendered pose of the robot.

Traversing this loss to find minima therefore enables the depicted robot pose to be estimated.

The afforementioned 3D Rendering is done via [Pyrender](https://github.com/mmatl/pyrender), with forward kinematics handled by [Klamp't](https://github.com/krishauser/Klampt).



# Installation

## Installing System-Wide Dependencies

This requires [Tensorflow](https://github.com/tensorflow/tensorflow) for both segmentation and pose estimation.

CUDA and cuDNN should first be installed according to the [Tensorflow Installation Instructions](https://www.tensorflow.org/install).

The recommended versions are:

[CUDA 11.0](https://developer.nvidia.com/cuda-11.0-download-archive), [cuDNN 8.0.4](https://developer.nvidia.com/rdp/cudnn-archive)

It is recommended to **not** install Visual Studio integration with CUDA (do a custom install and deselect VS integration).

## Final Installation

It is recommended to simply install requirements with pip:
```bash
pip install --upgrade -r requirements.txt
```

### Issues That May Occur on Install

<details>
  <summary> Keras </summary>

Keras sometimes includes ```.decode('utf8')``` in its code. This is unnecessary, and causes issues when loading and saving hd5f files.

Notably, every instance of ```.decode('utf8')``` in ```"tensorflow_core\python\keras\saving\hdf5_format.py"``` can be removed.

This will often cause issues when loading a model for segmentation.

</details>

<details>
  <summary> Numpy </summary>

**This is avoided if using ```requirements.txt``` to install**

Numpy **1.19.5** may be automatically installed with Tensorflow. This version of Numpy presents memory issues on some machines when utilizing the multiprocessing module, as this repository does.

Numpy **1.19.2** should work with this repository.

</details>

# Customization

Many static constants are specified in [robotpose/constants.py](robotpose/constants.py). Many of these can be easily modified to change program behavior.

Some constants only change visual appearance of some functions whereas others can change how the entire program behaves; take caution whenever changing more technical constants.

# License

Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/deepposekit/blob/master/LICENSE) for details.
