import numpy as np
import pickle
import json
from skimage import io
from pathlib import Path

from feature_extractors.extractors import Extractors
from utils.utilities import *
from indexer.IMD2020_Indexer import IMD2020_Indexer
from data.constants import FEATURE_EXTRACTORS

def Index2Features(
        dataset_index:dict,
        feature_extractors:list[str],
        patch_size=16,
        stride=8,
        mask_threshold=0.1
):
    X = []
    Y = []
    feat_dims = {}

    for key, path in dataset_index.items():

        print('Processing: ', Path(path['original']).stem)
        orig_img = io.imread(path['original'])

        orig_feature_list = []
        extr = Extractors(orig_img, patch_size, stride)
        for extractor in feature_extractors:

            if not hasattr(extr, extractor):
                raise NameError(
                    f"Extractor '{extractor}' is not a valid function."
                    f"Try using extractors in these list: {FEATURE_EXTRACTORS}"
                )
            
            method = getattr(extr, extractor)
            feat = method()
            orig_feature_list.append(feat)

            if extractor not in feat_dims:
                feat_dims[extractor] = feat.shape[1]

        orig_features = np.hstack(orig_feature_list)
        orig_labels = np.zeros(orig_features.shape[0], dtype=np.uint8)

        X.append(orig_features)
        Y.append(orig_labels)

        for sample in path['manipulated']:

            print('Processing: ', Path(sample['image']).stem)
            manip_img = io.imread(sample['image'])

            manip_feature_list = []
            extr = Extractors(manip_img, patch_size, stride)
            for extractor in feature_extractors:

                method = getattr(extr, extractor)
                feat = method()
                manip_feature_list.append(feat)

            manip_features = np.hstack(manip_feature_list)

            mask_img = io.imread(sample['mask'])
            manip_labels = mask2labels(
                mask_img, mask_threshold, patch_size, stride
            )

            if len(manip_labels) != manip_features.shape[0]:
                raise ValueError(
                    f"Patch count mismatch between manipulated image '{sample['image'].stem}' and its mask '{sample['mask'].stem}'"
                    f"The manipulated has {manip_features.shape[0]} patches while the mask has {len(manip_labels)}"
                )
            
            X.append(manip_features)
            Y.append(manip_labels)

    print('\n--- Feature Dimensions per Extractor ---')
    total = 0
    for extractor, dim in feat_dims.items():
        print(f'  {extractor}: {dim}')
        total += dim
    print(f'  Total: {total}\n')

    return X, Y

INDEX_PATH = 'data/json/imd2020_index.json'

with open(INDEX_PATH, 'r') as f:
    index_dict = json.load(f)

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

sample_dict = dict(list(index_dict.items())[310:350])
X, Y = Index2Features(sample_dict, feature_extractors, patch_size=32, stride=16)

with open('data/feat_matrix/X/X10.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('data/feat_matrix/Y/Y10.pkl', 'wb') as f:
    pickle.dump(Y, f)

print(len(X[0][0]), len(X), len(Y))