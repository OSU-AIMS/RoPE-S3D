# Datasets

Raw and compiled data is stored here.

Place raw zip data in /data/raw/ before compiling.

Compilation occurs automatically if loading of uncompiled dataset is attempted.

*datasets.json* will automatically update whenever any instance of the ```Dataset``` class is instantiated.

To use a dataset, use the Dataset class:
```python
from robotpose.dataset import Dataset
ds1 = Dataset('set2_sl') # Using full dataset name
ds2 = Dataset('set2')    # Using partial name
```

## Data Structure
Data should be arranged as follows:
```angular2html
├── ...
├── data
|   |── raw
|   |   |── set6_slu.zip             # Placed by user
|   |   |── set7_lu.zip              
|   |   └── ...
|   |  
|   |── set6_slu                     # Compilied by program
|   |   |── set6_slu.h5
|   |   |── og_vid.avi
|   |   └── 
|   |   
|   └── ...
└── ...
```

## Data Expectations

Data is expected to be placed in a ```.zip``` archive. 

It can be organized in subfolders, however these will be disregarded when compiling data, although their order may impact data order.

Associated files must have the same basename, only differing in extension.
---------------------------------
The following are expected:

| Extension  | Data type                         |
| ---------- | - |
| .png       | RGB image                         |
| .npy       | uint16 depthmap                   |
| .json      | JSON info dict (see format below) |

The JSON information file is expected to follow the following format and contain the following information at a minimum:
```json
{
  "objects": [
    {
      "joints": [
        {
          "angle": -1.0,
          "name": "link_1",
          "position": [
            0.0,
            0.0,
            0.0
          ]
        },
        {
          "angle": 1.5,
          "name": "link_2",
          "position": [
            0.04754660291639629,
            -0.0740494466630949,
            0.0
          ]
        }
      ]
    }
  ],
  "realsense_info":[
      {
        "depth_scale" : 0.0001,
        "intrin_depth": "depth_intrinsics_string",
        "intrin_color": "color_intrinsics_string"
      }
  ]
}

```