import numpy as np
import matplotlib.pyplot as plt
import os, sys, types
import joblib
import pickle
from time import time
from skimage import io
from pathlib import Path
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import RobustScaler
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_preprocessing')))
from data.constants import FEATURE_EXTRACTORS
from feature_extractors import Extractors
from src.normalizer import RobustScaler # type: ignore
from utils.utilities import *

def extract_features(
    image_path: str | None = None,
    feature_extractors: list[str] | None = None,
    patch_size=32,
    stride=16,
    *args
):
    start_time = time()

    print('Processing: ', Path(image_path).stem)
    img = io.imread(Path(image_path))
    
    h, w, _ = img.shape
    _, h, w = resize4patch(img, h, w, patch_size)
    patches = patchify(
        img,
        h, w,
        patch_size=patch_size,
        stride=stride
    )

    feat_list = []
    extr = Extractors(img, patch_size, stride, *args)
    for extractor in feature_extractors:

        if not hasattr(extr, extractor):
            raise NameError(
                f"Extractor '{extractor}' is not a valid function."
                f"Try using extractors in this list: {FEATURE_EXTRACTORS}"
            )
        
        method = getattr(extr, extractor)
        feat = method()
        feat_list.append(feat)

    features = np.array([])
    features = np.hstack(feat_list)

    end_time = time()
    wall_time = end_time - start_time
    print(f'Extraction Time: {wall_time:.4f}s')
    print(features[0].size)
    
    return features, tuple((patches, h, w))

feature_extractors = [
    'noise_residual_extract',
    'wavelet_extract',
    'fft_extract',
    'dct_extract',
    'lbp_extract',
    'glcm_extract',
    'stat_extract',
    'gsm_extract',
    'laplacian_stat_extract',
    'hprb_extract',
    'hog_extract',
    'color_hist_extract',
    'chroma_correlation_extract'
]

# IMG_PATH = Path('IMD2020/1al8bl/c8yfhte_0.jpg')
# IMG_PATH = Path('IMD2020/1a5x44/c8uefj0_0.jpg')
IMG_PATH = Path('IMD2020/1as9ik/c90ciwt_0.png')

PRE_PATH = Path('data/feat_matrix/processed')
SCALER_CHKP = Path('scaler.pkl')
FEAT_CHKP = Path('selected_cols.pkl')

CHKP_PATH = Path('training/checkpoints')
CB_CHKP = Path('catboost_checkpoint.cbm')
XGB_CHKP = Path('xgboost_checkpoint.json')
LGBM_CHKP = Path('lightgbm_checkpoint.joblib')
META_CHKP = Path('meta_learner.joblib')

threshold = 0.15

# ====================================
# LOAD CHECKPOINTS
# ====================================
print('Loading Checkpoints...')
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

# Load Selected Features
if os.path.exists(PRE_PATH / FEAT_CHKP):
    with open(PRE_PATH / FEAT_CHKP, 'rb') as f:
        selected_cols = pickle.load(f)
else:
    raise FileNotFoundError(f'Selected Features checkpoint does not exist at this path: "{PRE_PATH / FEAT_CHKP}"')

# Load Scaler
if os.path.exists(PRE_PATH / SCALER_CHKP):
    with open(PRE_PATH / SCALER_CHKP, 'rb') as f:
        scaler = pickle.load(f)
else:
    raise FileNotFoundError(f'Scaler checkpoint does not exist at this path: "{PRE_PATH / SCALER_CHKP}"')

print('Checkpoints Loaded Successfully!\n')

# ====================================
# FEATURE EXTRACTING AND PREPROCESSING
# ====================================
print('\nExtracting Features From The Image...')
img_feat, img_info = extract_features(IMG_PATH, feature_extractors)
img_feat = img_feat[:, selected_cols]
img_feat = scaler.transform(img_feat)

# ====================================
# PREDICT PROBABILITIES
# ====================================
print('\nPredicting...')
start = time()
cb_proba = cb.predict_proba(img_feat)[:, 1]
xgb_proba = xgb.predict_proba(img_feat)[:, 1]
lgbm_proba = lgbm.predict_proba(img_feat)[:, 1]
end = time()

# =======================================
# WEIGHTED VOTING (STACKING META-LEARNER)
# =======================================
print('\nEnsembling (Weighted Voting)...')
hard_start = time()

X_meta_test = np.column_stack((cb_proba, xgb_proba, lgbm_proba))
weighted_proba = meta_model.predict_proba(X_meta_test)[:, 1]

hard_end = time()
hard_inf_time = (end - start) + (hard_end - hard_start)
print(f'Weighted Voting Inference Time: {hard_inf_time:.4f}s')

# =======================================
# UNPATCHIFY
# =======================================
patches = np.array(img_info[0])
h, w = img_info[1:]

reconstructed_img = unpatchify(patches, h, w, patch_size=32, stride=16)
heatmap = unpatchify(weighted_proba, h, w, patch_size=32, stride=16)
heatmap_masked = np.ma.masked_where(heatmap < threshold, heatmap)

# =======================================
# VISUALIZE THE OVERLAY
# =======================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.imshow(reconstructed_img.astype(np.uint8)) 
ax1.set_title("Reconstructed Image")
ax1.axis('off')

divider1 = make_axes_locatable(ax1)
cax1 = divider1.append_axes("right", size="5%", pad=0.1)
cax1.axis('off')

ax2.imshow(reconstructed_img.astype(np.uint8))
im = ax2.imshow(heatmap_masked, cmap='gnuplot_r', alpha=0.5, vmin=0.15, vmax=1.0) 
ax2.set_title("Forgery Prediction Heatmap Overlay")
ax2.axis('off')

divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.1)
fig.colorbar(im, cax=cax)

plt.tight_layout(pad=1.0)
plt.show()