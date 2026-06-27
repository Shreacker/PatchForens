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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score

VAL_PATH = Path('data/feat_matrix/processed/val')
PROCESSED_PATH = Path('data/feat_matrix/processed')

X_VAL = Path('X_val.bin')
Y_VAL = Path('y_val.parquet')
TARGET = 'label'

CHKP_PATH = Path('training/checkpoints')
CB_CHKP = Path('catboost_checkpoint.cbm')
XGB_CHKP = Path('xgboost_checkpoint.json')
LGBM_CHKP = Path('lightgbm_checkpoint.joblib')

# ====================================
# LOAD DATA
# ====================================
if not os.path.exists(VAL_PATH / X_VAL) or not os.path.exists(VAL_PATH / Y_VAL):
    raise FileNotFoundError(f'Validation files are missing. (Current files: {os.listdir(VAL_PATH)})')

else:
    print('Loading Data...')
    y_val = pd.read_parquet(VAL_PATH / Y_VAL)
    
    val_rows = len(y_val)
    n_feats = 100

    X_val = np.memmap(
        VAL_PATH / X_VAL,
        dtype=np.float32,
        mode='r',
        shape=(val_rows, n_feats)
    )

    X_val = np.array(X_val)
    y_val = y_val[TARGET].values
    print('Done Loading!')

print('Dataset Size and Unique Classes:')
print(X_val.shape, y_val.shape)
print(np.unique(y_val))

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

print('\nModel Loaded Successfully!')

# ====================================
# PREDICT PROBABILITIES
# ====================================
print('\nPredicting...')
cb_proba = cb.predict_proba(X_val)[:, 1]
xgb_proba = xgb.predict_proba(X_val)[:, 1]
lgbm_proba = lgbm.predict_proba(X_val)[:, 1]

X_meta_train = np.column_stack((cb_proba, xgb_proba, lgbm_proba))

print(
    '=========================================\n'
    'Logistic Regression Meta-Learner Training\n'
    '========================================='
)
meta_model = LogisticRegression(random_state=42)
meta_model.fit(X_meta_train, y_val)

print("\nLearned Model Weights (Coefficients):")
print(f"CatBoost Weight:  {meta_model.coef_[0][0]:.4f}")
print(f"XGBoost Weight: {meta_model.coef_[0][1]:.4f}")
print(f"LightGBM Weight: {meta_model.coef_[0][2]:.4f}")

if True:
    joblib.dump(meta_model, CHKP_PATH / 'meta_learner.joblib')
    print('Saved Meta-Learner Model for Ensemble Weighting!')

if False:
    model = joblib.load(CHKP_PATH / 'meta_learner.joblib')