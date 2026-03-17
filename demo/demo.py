import numpy as np
import time
from skimage import io
from pathlib import Path

from feature_extractors import Extractors
from utils import *
from data.constants import FEATURE_EXTRACTORS

def demoImage(
    image_path:str,
    feature_extractors:list[str],
    patch_size=32,
    stride=16,
    mask_threshold=0.1,
    *args
):
    start_time = time.time()

    print('Processing: ', Path(image_path).stem)
    img = io.imread(Path(image_path))
    # img_edge = model(image_path)

    # Raw Image Feature Extraction
    feat_list = []
    extr = Extractors(img, patch_size, stride, *args)
    for extractor in feature_extractors:

        if not hasattr(extr, extractor):
            raise NameError(
                f"Extractor '{extractor}' is not a valid function."
                f"Try using extractors in these list: {FEATURE_EXTRACTORS}"
            )
        
        method = getattr(extr, extractor)
        feat = method()
        feat_list.append(feat)

    feature = np.hstack(feat_list)

    end_time = time.time()
    wall_time = end_time - start_time
    print(f'Wall time: {wall_time:.4f}s')
    print(feature[0].size)
    
    return None

def demoBatch():
    return None

feature_extractors = [
    'lbp_extract',
    'glcm_extract',
    'stat_extract',
    'noise_residual_extract',
    'wavelet_extract',
    'fft_extract',
]
IMG_PATH = 'demo/Avatar.jpg'
demoImage(IMG_PATH,
          feature_extractors,
          )