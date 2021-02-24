# Datasets

Raw and compiled data is stored here.

Place raw data in /data/raw/ before compiling.

To compile a dataset manually, use the following function:
```python
from robotpose.dataset import build
build(raw_data_path, compiled_data_path)
```

To use a dataset, use the Dataset class:
```python
from robotpose.dataset import Dataset
ds1 = Dataset('set2_sl') # Using full dataset name
ds2 = Dataset('set2')    # Using partial name
```
Loading a dataset that has not been compiled but is in the /data/raw/ directory will auto-compile the data for use.

## Data Structure
Data should be arranged as follows:
```angular2html
├── ...
├── data
|   |── raw
|   |   |── set3_lu                 # Placed by user
|   |   |   |──2021022300001.json
|   |   |   |──2021022300001.ply
|   |   |   |──2021022300001_og.png
|   |   |   |──2021022300001_rm.png
|   |   |   |──2021022300002.json
|   |   |   └── ...
|   |   └── ...
|   |  
|   |── set3_lu                     # Compilied by program
|   |   |── ds.json
|   |   |── ang.npy
|   |   |── og_img.npy
|   |   |── rm_img.npy
|   |   |── og_vid.avi
|   |   |── rm_vid.avi
|   |   |── ply.pyc
|   |   └── 
|   |   
|   |── skeletons                   # Created by user
|   |   |── A.csv
|   |   |── B.csv
|   |   └── ...
|   |   
|   └── ...
└── ...
```
