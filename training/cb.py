import numpy as np
import pandas as pd
import os, gc
import pickle
from time import time
from pathlib import Path
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

TRAIN_PATH = Path('data/feat_matrix/processed/train')
VAL_PATH = Path('data/feat_matrix/processed/val')
TEST_PATH = Path('data/feat_matrix/processed/test')
PROCESSED_PATH = Path('data/feat_matrix/processed')

X_TRAIN = Path('X_train.bin')
Y_TRAIN = Path('y_train.parquet')
X_VAL = Path('X_val.bin')
Y_VAL = Path('y_val.parquet')
X_TEST = Path('X_test.bin')
Y_TEST = Path('y_test.parquet')
TARGET = 'label'

CHKP_PATH = Path('training/checkpoints')

# ====================================
# LOAD DATA
# ====================================
if not os.path.exists(TRAIN_PATH / X_TRAIN) or not os.path.exists(TRAIN_PATH / Y_TRAIN):
    raise FileNotFoundError(f'Train files are missing. (Current file: {os.listdir(TRAIN_PATH)})')

elif not os.path.exists(VAL_PATH / X_VAL) or not os.path.exists(VAL_PATH / Y_VAL):
    raise FileNotFoundError(f'Validation files are missing. (Current files: {os.listdir(VAL_PATH)})')

elif not os.path.exists(TEST_PATH / X_TEST) or not os.path.exists(TEST_PATH / Y_TEST):
    raise FileNotFoundError(f'Test files are missing. (Current files: {os.listdir(TEST_PATH)})')

else:
    print('Loading Data...')
    y_train = pd.read_parquet(TRAIN_PATH / Y_TRAIN)
    y_val = pd.read_parquet(VAL_PATH / Y_VAL)
    y_test = pd.read_parquet(TEST_PATH / Y_TEST)
    
    train_rows = len(y_train)
    val_rows = len(y_val)
    test_rows = len(y_test)
    n_feats = 100

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

    X_test = np.memmap(
        TEST_PATH / X_TEST,
        dtype=np.float32,
        mode='r',
        shape=(test_rows, n_feats)
    )

    X_train = np.array(X_train)
    y_train = y_train[TARGET].values
    X_val = np.array(X_val)
    y_val = y_val[TARGET].values
    X_test = np.array(X_test)
    y_test = y_test[TARGET].values
    print('Done Loading!')

# Choosing top-k features from 100 features
top_k = 100
print(f'Take top-{top_k} features from the data')
X_train = X_train[:, :top_k]
X_val = X_val[:, :top_k]
X_test = X_test[:, :top_k]

print('Dataset Size and Unique Classes:')
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)
print(np.unique(y_train), np.unique(y_val), np.unique(y_test))

# ====================================
# MODEL STRUCTURE AND TRAINING
# ====================================
print('---------------------------------')
print('Initializing Model...')
model = CatBoostClassifier(
    grow_policy='Lossguide',
    iterations=10000,
    learning_rate=0.02,
    subsample=1.0,
    depth=12,
    max_leaves=140,
    min_data_in_leaf=5, # min_child_samples
    colsample_bylevel=0.65,
    l2_leaf_reg=3.0, # lambda
    loss_function='Logloss',
    eval_metric='AUC',
    custom_metric=['Logloss', 'PRAUC'],
    early_stopping_rounds=50,
    random_seed=42,
    thread_count=-1,
    verbose=200,
    train_dir='./temp/catboost_info/',
)

start = time()
print('Start Training Model...')
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    use_best_model=True
)
end = time()
train_time = (end - start) / 60.0
print(f'\nTraining Complete! Training Time: {train_time:.2f}')

print(
    '\n'
    '====================================\n'
    'EVALUATION ON VALIDATION SET\n'
    '===================================='
)
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]
con_mat = confusion_matrix(y_val, y_pred)

print('\n---Classification Report---')
print(classification_report(y_val, y_pred))
print(f'Accuracy Score: {accuracy_score(y_val, y_pred):.4f}')
print(f'ROC AUC Score: {roc_auc_score(y_val, y_pred_proba):.4f}')
print(f'Confusion Matrix:\n {con_mat}')

print(
    '\n'
    '====================================\n'
    'EVALUATION ON TEST SET\n'
    '===================================='
)
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)[:, 1]
test_con_mat = confusion_matrix(y_test, y_test_pred)

print('\n---Classification Report---')
print(classification_report(y_test, y_test_pred))
print(f'Accuracy Score: {accuracy_score(y_test, y_test_pred):.4f}')
print(f'ROC AUC Score: {roc_auc_score(y_test, y_test_pred_proba):.4f}')
print(f'Confusion Matrix:\n {test_con_mat}')

# ====================================
# SAVING MODEL CHECKPOINTS
# ====================================
if True:
    model.save_model(CHKP_PATH / 'catboost_checkpoint.cbm')

# ====================================
# LOAD MODEL CHECKPOINTS
# ====================================
if False:
    model = CatBoostClassifier()
    model.load_model(CHKP_PATH / 'catboost_checkpoint.cbm')