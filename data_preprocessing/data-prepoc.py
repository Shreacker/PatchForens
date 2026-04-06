import numpy as np
import pandas as pd
import pickle
import os
import tarfile
import gzip
from pathlib import Path

from sbss import SBSS

df_path = Path('../data/feat_matrix/Manipulate-Image-Features.pkl')
archive_path = df_path.with_name(df_path.stem + '.tar.gz')

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
    
sbss = SBSS(
    df,
    non_pca_cols=['image_id', 'label'],
    random_state=21
    )

X_train, X_test, y_train, y_test = sbss.train_test_split(test_size=0.2)