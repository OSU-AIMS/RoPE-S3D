# General Prediction

## Stages

Prediction is based upon the idea of *stages*.

Each stage simulates the robot in multiple positions, hoping to minimize a loss function. As input data is noisy and very detailed, this was deemed the best way to determine a robot's pose without using markers.

Stages are applied sucessively and can be fine-tuned to provide differing amounts of speed, precision, and accuracy.

Stage progressions can be modified in [robotpose/prediction/stages.py](robotpose/prediction/stages.py).

Stages are set by the function below:
```python
from robotpose.prediction.stages import *
getStages() # Where all stages are defined
```


## Available Stages

The following are the stages used in this repo, in order of decreasing complexity.

```python
from robotpose.prediction.stages import *

Lookup()
'''Compares input data to thousands of pre-rendered poses, 
selecting the closest using a simplified loss function with GPU acceleration.'''

Descent()
'''Compares current pose to poses with very small differences.
The magnitude of change of the joint angles decreases as this progresses.'''

InterpolativeSweep()
'''Compares the current pose to n poses with a certain joint
or set of joints in a range of positions.
Tries to interpolate loss to find a minima.'''
# Aliases
IntSweep()
ISweep()

TensorSweep()
'''Similar to ISweep, but uses the same simplified and accelerated 
loss function as Lookup(), although poses are rendered in realtime.'''
# Alias
TSweep()

SFlip()
'''Compares the current pose to one with a visually similar 
'shadow' that is often a relative minimum of the loss function.'''
```

### The Lookup Stage

The lookup stage is unique in that it relies upon data to be pre-rendered for comparison.

Because of this, lookup settings are static and are set in [robotpose/constants.py](robotpose/constants.py).

Additionally, the number of joints varying in a lookup can have significant effect on the performance of prediction, especially when running at high resolutions.

When running at low resolutions and/or with a limited range of motion, it may be favorable to do a lookup on ``SLU``. Higher resolutions or larger ranges of motion may favor better with an ```SL``` lookup.

**Note: As the lookup stage is a prerequisite for all other stages, care should be taken to re-adjust subsequent changes after modifying lookup settings**

### Descent and Sweep Animations on SLU
<img src="assets/descent.gif" width="250" />
<img src="assets/sweeps.gif" width="250" />

## Running Dataset Prediction

To predict on a dataset, change the 'dataset' variable hardcoded in ```predict.py```. Then run this script.

Average prediciton speed ranges from 0.5-2 seconds per pose. This may vary significally with computer specifications and input resolution.

<details>
  <summary> Note on Accuracy </summary>
    Running this script will provide results for all data in the dataset, regardless of if the segmentation model was trained on the data.
    To view predictions on those poses of the dataset that have not been used for segmentation training, it is advisable to split the data into multiple datasets (with the same camera pose) and to train on one and evalute performace with another.
</details>


# Live Prediction

Currently, live prediction has only be tested with the robot in static poses.

Reported pose data is retrieved from a `.json` file that is in a shared LAN folder.

To control the robotic arm the [RoPE Capture Tool](https://github.com/OSU-AIMS/RoPE-Capture-Tool) is used, which could be a good starting point for custom installs.

## Using JSON Link

`python predict_live.py` uses the afforementioned shared JSON strategy. This is only for proof of concept.

To use this, you will need to share a directory between the two systems being used.

This can be done by hosting the share on Linux using Samba (R Click -> Sharing Options), and connection to it via Windows (RUN -> `[linux_ip]\[share_name]`)

This file address should then be changed in [constants.py](robotpose/constants.py).

### Behavior

This approach will place the robot in a random position, wait for prediction to occur, then move the robot to a new position.

## Custom Usage

This repository requires the data from ROS' ```joint_states``` node to ascertain reported position.

This can be ingested in any way you see fit. See [textfile_integration.py](robotpose/testfile_integration.py) for a reference on how our JSON link setup works.

It would also be helpful to look through [predict_live.py](predict_live.py).
