from .wizards import Wizard
from .data import Dataset, DatasetInfo, AutomaticAnnotator
from .simulation import DatasetRenderer, RobotLookupCreator
from .prediction import Predictor, LiveCamera
from .paths import Paths
from .prediction.analysis import Grapher
from .prediction.synthetic import SyntheticPredictor
from .textfile_integration import JSONCoupling
from .projection import Intrinsics

import logging
logging.basicConfig(level=logging.INFO)

Paths().create()
