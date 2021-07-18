import robotpose
import numpy as np
from robotpose.utils import Grapher
from robotpose import Dataset

dataset = 'set21'


ds = Dataset(dataset)

preds = np.load(f'predictions_{dataset}.npy')
angles = np.copy(ds.angles)

errs = np.abs(preds - angles)
err = np.sum(errs,-1)

# preds = preds[err>3]
# angles = angles[err>3]




indicies = np.argsort(angles[...,0])

out = np.sort(indicies[:600])
# print(out)

g = Grapher('SLU',preds[indicies],angles[indicies])
g.plot()


