# RoPE-S3D

**Ro**botic **P**ose **E**stimation using **S**egmented **3**-**D**imensional images

This tool uses Artifical Intelligence (AI), specifically image segmentation algorithms provided by [PixelLib](https://github.com/ayoolaolafenwa/PixelLib), to predict robot joint angles.

This is acomplished by segmenting the robot and each of its joints from an RGBD (3-D) image. *With a known camera pose realtive to the robot*, these segmented sections of the image can be compared to a 3-D rendering of the robot in any possible pose.

An error function can be defined, quantitatively comparing these images based on visual and depth-based similarity. For any given input, this characterizes loss as a function of the rendered pose of the robot.

Traversing this loss to find minima therefore enables the depicted robot pose to be estimated.

The afforementioned 3D Rendering is done via [Pyrender](https://github.com/mmatl/pyrender), with forward kinematics handled by [Klamp't](https://github.com/krishauser/Klampt).

Depth visualization uses the [Turbo Colormap](https://ai.googleblog.com/2019/08/turbo-improved-rainbow-colormap-for.html).

# Usage

The following are the main steps that must be taken to train a new model:
1. Create a dataset
2. Align the dataset
3. Perform automatic annotation
4. Train the model

## Meshes

Meshes are loaded directly from a robot's URDF.

The active URDF can be changed by running the wizard (with ```python wizard.py```), which will automatically 

## Datasets

Datasets are expected to contain RGB images in a ```.png``` format with accompanying depthmaps in a ```.npy``` array file, and a ```.json``` information file.

To build, or recompile a dataset, simply run the wizard with arguments:
```bash
python wizard.py dataset_name [-rebuild] [-recompile]
```
With ```-rebuild``` recreating the dataset from the raw data directly, and with ```-recompile``` reprocessing the dataset from the raw data stored in the dataset itself.

## Automatic Annotation

### Alignment

Before running any automatic annotation, first align the dataset with the render using the Aligner found in the wizard: ```python wizard.py```

1. Select Dataset
2. Click "Align"

### Annotation

Then, use the automatic annotation script:

```bash
python annotate.py dataset_name [-no_preview]
```

## Training

```bash
python train.py dataset_name [--batch] [--valid]
```

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

## Known Issues

### Keras

Keras sometimes includes ```.decode('utf8')``` in its code. This is unnecessary, and causes issues when loading and saving hd5f files.

Notably, every instance of ```.decode('utf8')``` in ```"lib\site-packages\tensorflow_core\python\keras\saving\hdf5_format.py"``` can be removed.

This will often cause issues when loading a model for segmentation.

#### Numpy

**This is avoided if using ```requirements.txt``` to install**

Numpy **1.19.5** may be automatically installed with tensorflow. This version of Numpy presents memory issues on some machines when running Multiprocessing, as this repository does.

Numpy **1.19.2** should work with this repository.

#### PixelLib

**In the suggested configuration using *Tensorflow 2.4.1* this is avoided.**

**This is avoided if using ```requirements.txt``` to install**

Sometimes Pixellib will not work after all installations have been completed when using Tensorflow 2.0.0. To fix this error, upgrade and downgrade Tensorflow.

```bash
pip install --upgrade tensorflow-gpu
pip install --upgrade tensorflow-gpu==2.0.0
```

# License

Released under a Apache 2.0 License. See [LICENSE](https://github.com/jgraving/deepposekit/blob/master/LICENSE) for details.
