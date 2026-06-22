import numpy as np
import pandas as pd
import pickle
import os, gc
from tqdm import tqdm
from pathlib import Path

from src.splitter import SBSS
from src.filter import Filter, InformationGain, SymmetricUncertainty, MeanCombiner
from src.balancer import ImageUndersamplingBinary
from src.normalizer import RobustScaler
from utils.utilities import export_split_to_disk

POST_EXTRACT_PATH = Path('data/feat_matrix/raw/base')
RAW_PATH = Path('data/feat_matrix/raw/reduced')
PROCESSED_PATH = Path('data/feat_matrix/processed')
os.makedirs(POST_EXTRACT_PATH, exist_ok=True)
os.makedirs(RAW_PATH, exist_ok=True)
os.makedirs(PROCESSED_PATH, exist_ok=True)

X_PATH = sorted(Path(POST_EXTRACT_PATH / 'X/X').glob('*.pkl'))
Y_PATH = sorted(Path(POST_EXTRACT_PATH / 'Y/Y').glob('*.pkl'))

FEAT_DTYPE = np.float32
OUT_BIN = Path('feat_matrix.bin')
OUT_META = Path('meta.parquet')

'''
MERGE DATA FROM RAW FRAGMENTED FEATURE MATRICES
'''
if not os.path.exists(RAW_PATH / OUT_BIN) or not os.path.exists(RAW_PATH / OUT_META):
    n_feats = None
    patch_counts = []
    labels_list = []

    print('Merging data from raw fragmented feature matrices...')
    with open(RAW_PATH / OUT_BIN, 'wb') as f:
        n_images = 0
        for x_path, y_path in zip(X_PATH, Y_PATH):
            with open(x_path, 'rb') as fx:
                X_shard = pickle.load(fx)
            with open(y_path, 'rb') as fy:
                Y_shard = pickle.load(fy)

            for x, y in tqdm(zip(X_shard, Y_shard), total=len(X_shard),
                             desc=os.path.basename(x_path)):
                if n_feats is None:
                    n_feats = x.shape[1]
                
                f.write(np.ascontiguousarray(x, dtype=FEAT_DTYPE).tobytes())
                patch_counts.append(x.shape[0])
                labels_list.append(np.asarray(y, dtype=np.int8))
                n_images += 1

            del X_shard, Y_shard
            gc.collect()

    total_rows = sum(patch_counts)
    print(f'{total_rows:,} rows x {n_feats} feats')

    feat_mat = np.memmap(
        RAW_PATH / OUT_BIN,
        dtype=FEAT_DTYPE,
        mode='r',
        shape=(total_rows, n_feats)
    )
    image_id_arr = np.repeat(np.arange(n_images), patch_counts).astype(np.int32)
    label_arr = np.concatenate(labels_list)

    meta_df = pd.DataFrame({
        'image_id': image_id_arr,
        'label': label_arr
    })
    meta_df.to_parquet(RAW_PATH / OUT_META)

else:
    print('Loading Data...')
    meta_df = pd.read_parquet(RAW_PATH / OUT_META)
    total_rows = len(meta_df)
    n_feats = 637 # Hyperparameter

    feat_mat = np.memmap(
        RAW_PATH / OUT_BIN,
        dtype=np.float32,
        mode='r',
        shape=(total_rows, n_feats)
    )
    print('Done Loading!')

'''
DATA SPLITTER
'''
print(
    '========================\n'
    'SPLITTING DATA WITH SBSS\n'
    '========================'
)
sbss = SBSS(
    feat_mat,
    meta_df,
    pca_components=50,
    random_state=21,
    image_id='image_id',
    label_col='label'
)

train_idx, val_idx, test_idx, y_train, y_val, y_test = sbss.train_val_test_split(
                                                            val_frac=0.1, test_frac=0.1
                                                        )
print(f'Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_idx)}')

del sbss
gc.collect()

'''
UNDERSAMPLE NON-MANIPULATED PATCHES IN TRAINING SET
'''
print(
    '==========================\n'
    'UNDERSAMPLING TRAINING SET\n'
    '=========================='
)
balancer = ImageUndersamplingBinary()
train_idx_bal = balancer.fit_transform(
    indices=train_idx,
    meta_df=meta_df,
    ratio=3.0, # Try [4.0, 5.0, 6.0]
    keep_original_ratio=0.05
)
y_train_bal = meta_df.loc[train_idx_bal, 'label'].values

'''
FEATURE SELECTOR
'''
top_k = 100
print(
    '==============================\n'
    f'SELECTING TOP-{top_k} FEATURES\n'
    '=============================='
)

ig = InformationGain()
su = SymmetricUncertainty()
combiner = MeanCombiner(top_k=top_k)

filter = Filter(methods=[ig, su], combiner=combiner)
filter.fit(X=feat_mat, y=y_train_bal, indices=train_idx_bal, n_bins=30)

selected_cols = filter.selected_columns
print(f'Kept {len(selected_cols)} out of {n_feats} columns')

# Save selected features
with open(PROCESSED_PATH / 'selected_cols.pkl', 'wb') as f:
    pickle.dump(selected_cols, f)

'''
DATA NORMALIZER
'''
print(
    '===================\n'
    'FIT SCALING DATASET\n'
    '===================='
)
scaler = RobustScaler()
scaler.fit(feat_mat, indices=train_idx_bal)

scaler.median = scaler.median[selected_cols]
scaler.IQR = scaler.IQR[selected_cols]

# Save scaler
with open(PROCESSED_PATH / 'scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

'''
SAVE PROCESSED DATA
'''
print(
    '========================\n'
    'SAVING PROCESSED DATASET\n'
    '========================'
)
for i in ['train', 'val', 'test']:
    os.makedirs(PROCESSED_PATH / i, exist_ok=True)

export_split_to_disk(
    X_source=feat_mat,
    meta_df=meta_df,
    row_indices=train_idx_bal,
    selected_cols=selected_cols,
    scaler=scaler,
    out_x_path=(PROCESSED_PATH / 'train' / 'X_train.bin'),
    out_y_path=(PROCESSED_PATH / 'train' / 'y_train.parquet')
)

export_split_to_disk(
    X_source=feat_mat,
    meta_df=meta_df,
    row_indices=val_idx,
    selected_cols=selected_cols,
    scaler=scaler,
    out_x_path=(PROCESSED_PATH / 'val' / 'X_val.bin'),
    out_y_path=(PROCESSED_PATH / 'val' / 'y_val.parquet')
)

export_split_to_disk(
    X_source=feat_mat,
    meta_df=meta_df,
    row_indices=test_idx,
    selected_cols=selected_cols,
    scaler=scaler,
    out_x_path=(PROCESSED_PATH / 'test' / 'X_test.bin'),
    out_y_path=(PROCESSED_PATH / 'test' / 'y_test.parquet')
)