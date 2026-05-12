import numpy as np
import pandas as pd
import pickle
import os
import tarfile
import gzip
from pathlib import Path

from sbss import SBSS

def robustScale(
        dataF: pd.DataFrame,
        cols: np.ndarray,
        train: bool=True,
        median: float=None,
        IQR=None,
        threshold: float=1e-2
    ):
    df = dataF.copy()

    if not train:
        if median is None and IQR is None:
            raise ValueError('Median and IQR of train set does not exist.')
        else:
            df[cols] = (df[cols] - median) / IQR

        return df
    
    else:
        Q1 = df[cols].quantile(0.25)
        Q3 = df[cols].quantile(0.75)
        IQR = Q3 - Q1
        IQR = IQR.clip(lower=threshold)
        median = df[cols].median()
        df[cols] = (df[cols] - median) / IQR

        return df, median, IQR

BASE_DIR = Path(__file__).resolve().parent
df_path = (BASE_DIR / '../data/feat_matrix/Manipulate-Image-Features.pkl').resolve()
archive_path = df_path.with_name(df_path.stem + '.tar.gz')

# Import data
if not os.path.exists(archive_path):
    with tarfile.open(archive_path, 'w:gz') as tar:
        try:
            tar.add(df_path, arcname=df_path.name)
        except FileNotFoundError as f:
            print('Cannot find the .pkl file or it doesn\'t exist')
else:
    with tarfile.open(archive_path, 'r:gz') as tar:
        pkl_members = [m for m in tar.getmembers()
                       if m.name.endswith('.pkl') and m.isfile()]
        
        if not pkl_members:
            raise ValueError('No .pkl file found in archive')
        
        member = pkl_members[0]
        print(f'Extracting: {member.name!r} size={member.size} bytes')

        if member.size == 0:
            raise ValueError('Pkl member has 0 bytes -- archive is corrupt, delete and re-run')
        
        f = tar.extractfile(member)
        if f is None:
            raise ValueError('Could not extract .pkl file')
            
        data = f.read()

    if data[:2] == b'\x1f\x8b':
        data = gzip.decompress(data)

    df = pickle.loads(data)

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f'Expected DataFrame, got {type(df)}')
    
# Feature Selection


# Splitting Data
sbss = SBSS(
    df,
    non_pca_cols=['image_id', 'label'],
    random_state=21
)
X_train, X_test, y_train, y_test = sbss.train_test_split(test_size=0.2)

# Normalization
feat_cols = X_train.filter(like='feat_').columns
X_train_norm, median, IQR = robustScale(X_train, feat_cols, train=True)
X_test_norm = robustScale(X_test, feat_cols, train=False, median=median, IQR=IQR)
