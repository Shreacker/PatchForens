import numpy as np
import pandas as pd
import os, gc
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

TRAIN_PATH = Path('data/feat_matrix/processed/train')
VAL_PATH = Path('data/feat_matrix/processed/val')

X_TRAIN = Path('X_train.bin')
Y_TRAIN = Path('y_train.parquet')
X_VAL = Path('X_val.bin')
Y_VAL = Path('y_val.parquet')
TARGET = Path('label')

CHKP_PATH = Path('training/checkpoints')

# ====================================
# LOAD DATA
# ====================================
if not os.path.exists(TRAIN_PATH / X_TRAIN) or not os.path.exists(TRAIN_PATH / Y_TRAIN):
    raise FileNotFoundError(f'Train files are missing. (Current file: {os.listdir(TRAIN_PATH)})')

elif not os.path.exists(VAL_PATH / X_VAL) or not os.path.exists(VAL_PATH / Y_VAL):
    raise FileNotFoundError(f'Train files are missing. (Current files: {os.listdir(VAL_PATH)})')

else:
    print('Loading Data...')
    y_train = pd.read_parquet(TRAIN_PATH / Y_TRAIN)[TARGET].values
    y_val = pd.read_parquet(VAL_PATH / Y_VAL)[TARGET].values
    
    train_rows = len(y_train)
    val_rows = len(y_val)
    n_feats = 100 # Hyperparameter

    X_train = np.memmap(
        TRAIN_PATH / X_TRAIN,
        dtype=np.float32,
        mode='r',
        shape=(train_rows, n_feats)
    )
    X_val = np.memmap(
        VAL_PATH / X_VAL,
        dtype=np.float32,
        mode='r',
        shape=(val_rows, n_feats)
    )

    X_train = np.array(X_train)
    X_val = np.array(X_val)
    print('Done Loading!')

print('Dataset Size:')
print(X_train.shape, len(y_train), X_val.shape, len(y_val))

# ====================================
# MODEL STRUCTURE AND TRAINING
# ====================================
print('---------------------------------')
print('Initializing Model...')
model = CatBoostClassifier(
    iterations=4000,
    learning_rate=0.05,
    subsample=1.0,
    depth=6,
    min_data_in_leaf=1, # min_child_samples
    colsample_bylevel=1.0,
    l2_leaf_reg=1.0, # lambda
    scale_pos_weight=1.0,
    loss_function='Logloss',
    eval_metric='PRAUC',
    custom_metric=['Logloss', 'AUC'],
    early_stopping_rounds=50,
    random_seed=42,
    thread_count=-1,
    verbose=200
)

print('Start Training Model...')
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    use_best_model=True
)

# ====================================
# MODEL EVALUATION
# ====================================
print('\nTraining Complete! Evaluating on Validation Set...')
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]

print('\n---Classification Report---')
print(classification_report(y_val, y_pred))
print(f'ROC AUC Score: {roc_auc_score(y_val, y_pred_proba):.4f}')

# ====================================
# SAVING MODEL CHECKPOINTS
# ====================================
if False:
    model.save_model(CHKP_PATH / 'catboost_checkpoint.cbm')

# ====================================
# LOAD MODEL CHECKPOINTS
# ====================================
if False:
    model = CatBoostClassifier()
    model.load_model("catboost_checkpoint.json")