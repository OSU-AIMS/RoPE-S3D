import argparse
import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # or any {'0', '1', '2'}
import tensorflow as tf

from robotpose.dataset import Dataset, DatasetInfo
from robotpose.wizards import DatasetWizard



def run(dataset, recompile, rebuild):
    ds = Dataset(dataset, recompile=recompile, rebuild=rebuild)



if __name__ == "__main__":

    if len(sys.argv) > 1:
        ds_info = DatasetInfo()
        ds_info.get()
        parser = argparse.ArgumentParser()
        parser.add_argument("dataset", type=str, default=None, choices=ds_info.unique_sets(), help="Dataset to run wizard on")
        parser.add_argument("-recompile", action='store_true', help="Reprocessing dataset from data stored in dataset")
        parser.add_argument("-rebuild", action='store_true', help="Recreate dataset from the raw data")
        args = parser.parse_args()
        run(args.dataset, args.recompile, args.rebuild)
    else:
        wiz = DatasetWizard()
        wiz.run()