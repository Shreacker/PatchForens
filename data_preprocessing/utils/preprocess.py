import numpy as np
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '../')
sys.path.insert(0, '.../')
from utilities import IG_SU
from sbss import SBSS
from ...utils.dataloader import Dataset

def filter(
        df: pd.DataFrame,
        x: list,
        y: str ='label'
):

    STDV = df.std()
    junk_cols = STDV[STDV < 1e-3].index
    df = df.drop(columns=junk_cols, axis=1)

    x = df.filter(like='feat_').columns
    keep_features, ranks, ig_su = IG_SU(df, x, y, keep_ratio=0.75)
    df = df[keep_features]

    # Correlation Filtering
    x = df.filter(like='feat_').columns
    return x

class Filter:
    def __init__(self):
        self.keep_feat = []
    
    def fit(self, ds):
        X, y = ds[:]

        STDV = ds.x.std()
        junk_cols = STDV[STDV < 1e-3].index
        X = X.drop(columns=junk_cols, axis=1)

def train_test_split(
        df: pd.DataFrame,
        non_pca_cols=['image_id', 'label'],
        random_state=21
):
    sbss = SBSS(
        df,
        non_pca_cols=non_pca_cols,
        random_state=random_state
    )

    X_train, X_test, y_train, y_test = sbss.train_test_split(test_size=0.2)

    return X_train, X_test, y_train, y_test

def robustScale(
        df: pd.DataFrame,
        x: list,
        threshold: float = 1e-2
):
    Q1 = df[x].quantile(0.25)
    Q3 = df[x].quantile(0.75)
    IQR = Q3 - Q1
    IQR = IQR.clip(lower=threshold)
    median = df[x].median()
    df[x] = (df[x] - median) / IQR

    return df, median, IQR

def RS_transform(
        df: pd.DataFrame,
        x: np.ndarray,
        median=None,
        iqr=None
):
    if median is None or iqr is None:
        raise ValueError
    
    df[x] = (df[x] - median) / iqr

    return df

def undersampling_per_image(
        X,
        y,
        image_id='image_id',
        threshold=3,
        random_state=21
):
    def func(x):
        df_non_manip = x[y.loc[x.index] == 0]
        df_manip = x[y.loc[x.index] == 1]
        manip_counts = len(df_manip)

        if len(df_manip) == 0:
            return pd.DataFrame()
        
        n_sample = min(int(manip_counts * threshold), len(df_non_manip))
        df_non_manip_reduced = df_non_manip.sample(n=n_sample, random_state=random_state)
        return pd.concat([df_manip, df_non_manip_reduced], axis=0).sample(frac=1)
    
    X_bal = X.groupby(image_id, observed=True, group_keys=False)\
            .apply(func, include_groups=False)
    
    y_bal = y.loc[X_bal.index]

    return X_bal.reset_index(drop=True), y_bal.reset_index(drop=True)