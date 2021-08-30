# Prediction

## Stages

Prediction is based upon the idea of *stages*.

Each stage simulates the robot in multiple positions, hoping to minimize a loss function. As input data is noisy and very detailed, this was deemed the best way to determine a robot's pose without using markers.

Stages are applied sucessively and can be fine-tuned to provide differing amounts of speed, precision, and accuracy.

### Available Stages

The following are the stages used in this repo, in order of decreasing complexity.

```python
from robotpose.prediction.stages import *

Lookup()    # Compares input data to thousands of pre-rendered poses, selecting the closest using a simplified loss function with GPU acceleration.

Descent()   # Compares current pose to poses with very small differences. The magnitude of change of the joint angles decreases as this progresses.

InterpolativeSweep()    # Compares the current pose to n poses with a certain joint or set of joints in a range of positions. Tries to interpolate loss to find a minima.
if True:    # Aliases
    IntSweep()
    ISweep()

TensorSweep()   # Similar to ISweep, but uses the same simplified and accelerated loss function as Lookup(), although poses are rendered in realtime.

SFlip() # Compares the current pose to one with a visually similar 'shadow' that is often a relative minimum of the loss function.
```
