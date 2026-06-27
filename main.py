import numpy as np
import pandas as pd
import os, gc
import pickle
import joblib
import lightgbm as lgb
from time import time
from pathlib import Path
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

TEST_PATH = Path('data/feat_matrix/processed/test')
PROCESSED_PATH = Path('data/feat_matrix/processed')

X_TEST = Path('X_test.bin')
Y_TEST = Path('y_test.parquet')
TARGET = 'label'

CHKP_PATH = Path('training/checkpoints')
CB_CHKP = Path('catboost_checkpoint.cbm')
XGB_CHKP = Path('xgboost_checkpoint.json')
LGBM_CHKP = Path('lightgbm_checkpoint.joblib')
META_CHKP = Path('meta_learner.joblib')

# ====================================
# LOAD DATA
# ====================================
if not os.path.exists(TEST_PATH / X_TEST) or not os.path.exists(TEST_PATH / Y_TEST):
    raise FileNotFoundError(f'Test files are missing. (Current files: {os.listdir(TEST_PATH)})')

else:
    print('Loading Data...')
    y_test = pd.read_parquet(TEST_PATH / Y_TEST)
    
    test_rows = len(y_test)
    n_feats = 100

    X_test = np.memmap(
        TEST_PATH / X_TEST,
        dtype=np.float32,
        mode='r',
        shape=(test_rows, n_feats)
    )

    X_test = np.array(X_test)
    y_test = y_test[TARGET].values
    print('Done Loading!')

print('Dataset Size and Unique Classes:')
print(X_test.shape, y_test.shape)
print(np.unique(y_test))

# ====================================
# LOAD CHECKPOINTS
# ====================================
# Load CatBoost Checkpoint
if os.path.exists(CHKP_PATH / CB_CHKP):
    cb = CatBoostClassifier()
    cb.load_model(CHKP_PATH / CB_CHKP)
else:
    raise FileNotFoundError(f'CatBoost checkpoint does not exist at this path: "{CHKP_PATH / CB_CHKP}"')

# Load XGBoost Checkpoint
if os.path.exists(CHKP_PATH / XGB_CHKP):
    xgb = XGBClassifier()
    xgb.load_model(CHKP_PATH / "xgboost_checkpoint.json")
else:
    raise FileNotFoundError(f'XGBoost checkpoint does not exist at this path: "{CHKP_PATH / XGB_CHKP}"')

# Load LightGBM Checkpoint
if os.path.exists(CHKP_PATH / LGBM_CHKP):
    lgbm = joblib.load(CHKP_PATH / LGBM_CHKP)
else:
    raise FileNotFoundError(f'LightGBM checkpoint does not exist at this path: "{CHKP_PATH / LGBM_CHKP}"')

# Load Meta-Learner Checkpoint
if os.path.exists(CHKP_PATH / META_CHKP):
    meta_model = joblib.load(CHKP_PATH / META_CHKP)
else:
    raise FileNotFoundError(f'Meta-Learner checkpoint does not exist at this path: "{CHKP_PATH / META_CHKP}"')

print('\nModel Loaded Successfully!')

# ====================================
# PREDICT PROBABILITIES
# ====================================
print('\nPredicting...')
start = time()
cb_proba = cb.predict_proba(X_test)[:, 1]
xgb_proba = xgb.predict_proba(X_test)[:, 1]
lgbm_proba = lgbm.predict_proba(X_test)[:, 1]
end = time()

# ====================================
# SOFT VOTING (UNWEIGHTED)
# ====================================
print('\nEnsembling (Soft Voting)...')
soft_start = time()

ensemble_proba = (cb_proba + xgb_proba + lgbm_proba) / 3.0

soft_end = time()
soft_inf_time = (end - start) + (soft_end - soft_start)
print(f'Soft Voting Inference Time: {soft_inf_time:.4f}')

# =======================================
# WEIGHTED VOTING (STACKING META-LEARNER)
# =======================================
print('\nEnsembling (Weighted Voting)...')
hard_start = time()

X_meta_test = np.column_stack((cb_proba, xgb_proba, lgbm_proba))
weighted_proba = meta_model.predict_proba(X_meta_test)[:, 1]

hard_end = time()
hard_inf_time = (end - start) + (hard_end - hard_start)
print(f'Weighted Voting Inference Time: {hard_inf_time:.4f}')

# ====================================
# EVALUATION WITH SOFT VOTING
# ====================================
print(
    '====================================\n'
    'SOFT ENSEMBLE EVALUATION ON TEST SET\n'
    '===================================='
)
threshold = 0.5
ensemble_pred = (ensemble_proba >= threshold).astype(int)
con_mat = confusion_matrix(y_test, ensemble_pred)

print('\n---Classification Report---')
print(classification_report(y_test, ensemble_pred))
print(f'Accuracy Score: {accuracy_score(y_test, ensemble_pred):.4f}')
print(f'ROC AUC Score: {roc_auc_score(y_test, ensemble_proba):.4f}')
print(f'Confusion Matrix:\n {con_mat}')

# ====================================
# EVALUATION WITH WEIGHTED VOTING
# ====================================
print(
    '\n'
    '========================================\n'
    'WEIGHTED ENSEMBLE EVALUATION ON TEST SET\n'
    '========================================'
)
threshold = 0.15
weighted_pred = (weighted_proba >= threshold).astype(int)
con_mat = confusion_matrix(y_test, weighted_pred)

print(f'\n---Classification Report---')
print(classification_report(y_test, weighted_pred))
print(f'Accuracy Score: {accuracy_score(y_test, weighted_pred):.4f}')
print(f'ROC AUC Score: {roc_auc_score(y_test, weighted_proba):.4f}')
print(f'Confusion Matrix:\n {con_mat}')