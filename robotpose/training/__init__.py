from .models import ModelManager, ModelInfo

import os
import requests

from tqdm import tqdm

from ..paths import Paths as p

if not os.path.isfile(p().BASE_MODEL):
    with tqdm(total=1, leave=False,desc='Downloading Base Model') as pbar:
        url = "https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.2/mask_rcnn_coco.h5"
        r = requests.get(url, allow_redirects=True)
        open(p().BASE_MODEL, 'wb').write(r.content)
        pbar.update(1)