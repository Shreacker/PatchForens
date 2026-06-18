import sys
from abc import ABC, abstractmethod

sys.path.insert(0, '../')
from ..utils.dataset import Dataset

class BaseNormalizer(ABC):
    @abstractmethod
    def fit(
            self,
            ds: Dataset
    ):
        ...

    @abstractmethod
    def transform(
            self,
            ds: Dataset
    ) -> Dataset:
        ...

class RobustScale(BaseNormalizer):
    def __init__(self):
        self.median = None
        self.IQR = None

    def fit(
            self,
            ds: Dataset,
            eps: float | None = 0.0
    ):
        X, y = ds[:]

        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)

        self.IQR = Q3 - Q1
        scale = X.abs().median()
        self.median = X.median()
        self.IQR = max(self.IQR, eps * scale)

        return self
    
    def transform(
            self,
            ds: Dataset
    ):
        if self.median is None or self.IQR is None:
            raise ValueError
        
        X, y = ds[:]
        X = (X - self.median) / self.IQR

        return Dataset(X, y)
    
class StandardScaler(BaseNormalizer):
    def __init__(self):
        self.mean = None
        self.STDV = None

    def fit(
            self,
            ds: Dataset,
            eps: float | None = None
    ):
        X, y = ds[:]

        self.mean = X.mean()
        self.STDV = X.std()
        self.STDV = max(self.STDV, eps)

        return self
    
    def transform(
            self,
            ds: Dataset
    ):
        X, y = ds[:]

        X = (X - self.mean) / self.stdv

        return Dataset(X, y)