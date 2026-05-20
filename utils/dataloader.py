import numpy as np
from torch.utils.data import DataLoader as _DataLoader

class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]
    
class DataLoader(_DataLoader):
    pass