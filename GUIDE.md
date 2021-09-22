# Installation

## Installing System-Wide Dependencies

This requires [Tensorflow](https://github.com/tensorflow/tensorflow) for both segmentation and pose estimation.

CUDA and cuDNN should first be installed according to the [Tensorflow Installation Instructions](https://www.tensorflow.org/install).

The recommended versions are: [CUDA 11.0](https://developer.nvidia.com/cuda-11.0-download-archive), [cuDNN 8.0.4](https://developer.nvidia.com/rdp/cudnn-archive)

It is recommended to **not** install Visual Studio integration with CUDA (do a custom install and deselect VS integration).

## Installing Python

This repository is only tested to work with **Python 3.6.4**.

It is advisable to create a virtual environment (using something like [Anaconda](https://www.anaconda.com/)) with this version of Python before continuing.

## Final Installation

Install requirements with pip:
```bash
pip install --upgrade -r requirements.txt
```

## Known Issues

<details>
  <summary> Keras </summary>
    Keras sometimes includes ```.decode('utf8')``` in its code. This is unnecessary, and causes issues when loading and saving hd5f files.

    Notably, every instance of ```.decode('utf8')``` in ```tensorflow_core\python\keras\saving\hdf5_format.py``` can be removed.

    This will often cause issues when loading a model for segmentation.

</details>



# Collect Data

**When collecting data it is highly recommended to obtain a rough idea of the camera's position relative to the robot's origin.**

## Expectations

Datasets are expected to contain RGB images in a ```.png``` format with accompanying depthmaps in a ```.npy``` array file, and a ```.json``` information file.

Each set of files should represent a single pose. If any errors occured in collection (if a pose could not be reached), single poses may be removed later using the **Verifier** in the **Wizard**.

The format for the ```.json``` info file can be found in [examples/dataset_json_required.json](examples/dataset_json_required.json).

## Collection

Collection can be done in numerous ways depending on your specific setup.

A starting point for collection can be obtained from our [RoPE Capture Tool](https://github.com/OSU-AIMS/RoPE-Capture-Tool), which is used to collect data from anInterl Realsense 435i and a Yaskawa MH5L on ROS Melodic.

### Pose Planner

In order to cover the entire state space of the robot uniformly, it is recommended to generate poses using a grid sampling method with or without noise.

A pose generator is included in this repo and operates on the actiove URDF, although it may not be suitable for all collection uses. It generates a ```.npy``` file of SLURBT poses to collect.

```bash
python collection_planner.py [-num MAX_POSES] [-file FILE_NAME] [-angs JOINTS] [-noise POSE_NOISE]
```

## Ingest

To load data into the repo, first collect all data into a ```.zip``` file. This should then be plased in [data/raw/](data/raw/).

The data will be referenced by the basename (filename less ```.zip```) of this file.

Then build the dataset with the wizard:
```bash
python wizard.py dataset_name [-rebuild]
```
```-rebuild``` can be used to recreate the dataset from the raw data directly, with the exception of stored camera poses, if later desired.


# Load a URDF

## Files

Place a robot's *entire* URDF in ```urdfs/```. This includes all referenced meshes.

Only ```.urdf``` files can be opened, ```.xacro``` files will noy be processed.

## In Wizard

Next, run ```python wizard.py```.

Select the URDF Tab. Make sure the desired URDF is selected. If it does not show up, a ```.urdf``` file cannot be found in [/urdfs/](/urdfs/).

# Align Camera

In order to automatically label data, the camera position relative to the robot must be recorded.

To do this, run ```python wizard.py```. Select the desired dataset. Click **Align**.

This will open the Aligner, where camera position can be tweaked. It is made of two windows.

The GUI allows for input of a pose (6-element list) and instructions for using the keyboard controls.

# Verify Poses

This allows for improperly-recorded poses to be removed from the dataset.

To do this, run ```python wizard.py```. Select the desired dataset. Click **Verify**.

**Depending on your camera settings, you may need to change the associated constants for rows/columns/scale in [robotpose/constants.py](robotpose/constants.py).**

Go through each pose, selecting improper poses. Selected images appear darker.

Confirm these poses to remove them from the dataset.

# Automatic Annotation

Run the automatic annotation script:

```bash
python annotate.py dataset_name [-no_preview]
```

# Configure Data Split

Data can be spit according to any desired ratios.

To do this, run ```python wizard.py```. Select the Training tab. Use the sliders to change ratios, and then click update.

# Training

Run the following script to train a model.

```bash
python train.py dataset_name [-batch_size size] [-cont] [-cont_from dataset_name]
```

Batch size can be adjusted (default 2) using -batch_size.

Using -cont will continue traininig the most recent model for this dataset.

Using -cont_from can be used to build a model off of a model from another dataset, which is recommended if possible.


# Prediction

**Please read [PREDICTION.md](PREDICTION.md) for specific instructions and details**

Prediction settings are more complex than other settings in this repo.

To change prediciton stages, the [stages.py](robotpose/prediction/stages.py) file in the prediction module must be modified.

## Dataset Prediction

To predict on a dataset, change the 'dataset' variable hardcoded in ```predict.py```. Then run this script.

Average prediciton speed ranges from 0.5-2 seconds per pose. This may vary significally with computer specifications and input resolution.

<details>
  <summary> Note on Accuracy </summary>
    Running this script will provide results for all data in the dataset, regardless of if the segmentation model was trained on the data.
    To view predictions on those poses of the dataset that have not been used for segmentation training, it is advisable to split the data into multiple datasets (with the same camera pose) and to train on one and evalute performace with another.
</details>

## Live Prediction

Live prediction involves more setup than dataset prediction. Please see [PREDICTION.md](PREDICTION.md) for specific instructions.


