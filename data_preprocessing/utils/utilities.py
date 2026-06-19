import numpy as np
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

sys.path.insert(0, '.../')
from .dataset import Dataset

def _entropy(x: pd.Series):
    if len(x) == 0:
        return 0
    
    val = x.value_counts(normalize=True).values

    return -np.sum(val * np.log2(val + 1e-9))

def _ig(ds: Dataset):
    X, y = ds[:]

    ent_Y = _entropy(y)
    ent_YX = 0.

    for x in np.unique(X):
        mask = X == x
        group_y = y[mask]
        ent_YX += (len(group_y)/len(y)) * _entropy(group_y)

    return ent_Y - ent_YX

def _su(ds: Dataset):
    ig_xy = _ig(ds)

    X, y = ds[:]

    ent_X = _entropy(X)
    ent_Y = _entropy(y)

    if (ent_X + ent_Y) < 1e-6:
        return 0
    
    return 2 * ig_xy / (ent_X + ent_Y)