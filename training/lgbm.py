import numpy as np
import pandas as pd
from pathlib import Path
from lightgbm import LGBMClassifier

TRAIN_PATH = 'data/feat_matrix/processed/train'
VAL_PATH = 'data/feat_matrix/processed/val'

CHKP_PATH = Path('training/checkpoints')