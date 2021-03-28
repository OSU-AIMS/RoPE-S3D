# Datasets

Raw and compiled data is stored here.

Place raw zip data in /data/raw/ before compiling.

Compilation occurs automatically if loading of uncompiled dataset is attempted.

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
|   |   |── set6_slu.zip             # Placed by user
|   |   |── set7_lu.zip              
|   |   └── ...
|   |  
|   |── set6_slu                     # Compilied by program
|   |   |── set6_slu.h5
|   |   |── set6_slu_train.h5
|   |   |── set6_slu_validate.h5
|   |   |── set6_slu_test.h5
|   |   |── og_vid.avi
|   |   |── seg_vid.avi
|   |   └── 
|   |   
|   |── skeletons                   # Created by user
|   |   |── A.csv
|   |   |── B.csv
|   |   |── A.json
|   |   |── B.json
|   |   └── ...
|   |   
|   └── ...
└── ...
```
