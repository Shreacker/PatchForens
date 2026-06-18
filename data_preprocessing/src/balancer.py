import sys
import pandas as pd
from abc import ABC, abstractmethod

sys.path.insert(0, '../')
from ..utils.dataset import Dataset

class BaseBalancer(ABC):
    @abstractmethod
    def fit_transform(
        self,
        ds,
        ratio: float | None = None,
        **kwargs
    ) -> Dataset:
        ...

class ImageUndersamplingBinary(BaseBalancer):
    def fit_transform(
            self,
            ds,
            image_id: str | None = None,
            ratio: float | None = None,
            **kwargs
    ):
        X, y = ds[:]

        def func(x):
            df_0 = x[y.loc[x.index] == 0]
            df_1 = x[y.loc[x.index] == 1]
            count_1 = len(df_1)

            if count_1 == 0:
                return pd.DataFrame()
            
            n_sample = min(int(count_1 * ratio), len(df_0))
            df_0_reduced = df_0.sample(n=n_sample, **kwargs)
            return pd.concat([df_1, df_0], axis=0).sample(frac=1)
        
        X_bal = X.groupby(image_id, observed=True, group_keys=False)\
                .apply(func, include_groups=False)
        y_bal = y.loc[X_bal.index]

        return Dataset(X_bal.reset_index(drop=True), y_bal.reset_index(drop=True))