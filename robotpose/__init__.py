from .wizards import Wizard
from .data import Dataset, DatasetInfo, AutomaticAnnotator
from .simulation import DatasetRenderer, RobotLookupCreator
from .prediction import Predictor
from .paths import Paths
from .prediction.analysis import Grapher
from .prediction.synthetic import SyntheticPredictor

import logging
logging.basicConfig(level=logging.INFO)

Paths().create()
