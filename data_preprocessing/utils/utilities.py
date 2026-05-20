import numpy as np
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '.../')
from ...utils.dataloader import Dataset

class IG_SU:
    def __init__(self):
        self.ig_scores = []
        self.su_scores = []

    def _entropy(self, x: pd.Series):
        if len(x) == 0:
            return 0
        
        val = x.value_counts(normalize=True).values

        return -np.sum(val * np.log2(val + 1e-9))
    
    def _ig(self, ds):
        X, y = ds[:]

        ent_Y = self._entropy(y)
        ent_YX = 0.

        for x in np.unique(X):
            mask = X == x
            group_y = y[mask]
            ent_YX += (len(group_y)/len(y)) * self._entropy(group_y)

        return ent_Y - ent_YX
    
    def _su(self, ds, ig_xy=None):
        if ig_xy is None:
            ig_xy = self._ig(ds)

        X, y = ds[:]

        ent_X = self._entropy(X)
        ent_Y = self._entropy(y)

        if (ent_X + ent_Y) < 1e-6:
            return 0
        
        return 2 * ig_xy / (ent_X + ent_Y)
    
    def fit(self, ds, bins=10, keep_ratio=1.):
        X, y = ds[:]
        
        for col in X.columns:
            if (X[col].dtype.kind in 'bifc') and len(np.unique(X[col])) > bins:
                binned_x = pd.qcut(X[col], bins, duplicates='drop')
                X_0 = pd.Series({col: binned_x})
                ds_0 = Dataset(X_0, y)
            else:
                X_0 = pd.Series({col: X[col]})
                ds_0 = Dataset(X_0, y)

            ig = self._ig(ds_0)
            su = self._su(ds_0)
            self.ig_scores.append(ig)
            self.su_scores.append(su)

        avg_scores = [(i + s) / 2 for i, s in zip(self.ig_scores, self.su_scores)]
        ranks = np.argsort(avg_scores)[::-1]
        scores_dict = dict(zip(X.columns, avg_scores))
        scores_dict = {key: value for key,
                       value in sorted(scores_dict.items(), reverse=True,
                                       key=lambda item: item[1])}
        
        n_keep = int(keep_ratio * len(X.columns))
        self.scores = pd.Series(scores_dict)
        self.keep_features = [X.columns[i] for i in ranks[:n_keep]]
        self.final_ranks = np.full(len(X.columns), -1)
        mask = self.scores.index.isin(self.keep_features)
        self.scores = self.scores[mask]

        for i in range(n_keep):
            self.final_ranks[ranks[i]] = i

        return self.keep_features, self.final_ranks, self.scores
    
    def transform(self, ds):
        return Dataset(ds.x.loc[:, self.keep_features], ds.y)