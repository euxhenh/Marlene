import json
import logging
import os
from collections import Counter, defaultdict
from functools import partial
from itertools import zip_longest
from json import JSONEncoder
from pathlib import Path

import anndata
import gseapy as gp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm


class JsonNumpy(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


json_dump = partial(json.dump, cls=JsonNumpy)

mpl.rcParams['pdf.fonttype'] = 42

logger = logging.getLogger(__name__)

npi = np.intersect1d
npc = np.concatenate
npu = np.unique
npurc = partial(np.unique, return_counts=True)
npd = np.setdiff1d
npin = np.in1d

r5 = anndata.read_h5ad
