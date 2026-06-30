import numpy as np
import pandas as pd
import cv2
from skimage import io, color
from skimage.color import label2rgb
from skimage.util import img_as_ubyte
from scipy.stats import skew as np_skew, kurtosis as np_kurtosis

def mask2labels(
        mask_img,
        mask_threshold=0.1,
        patch_size=16,
        stride=8
):
    mask_img = convert2gray(mask_img, binary=True)
    h, w = mask_img.shape
    mask_img, h, w = resize4patch(mask_img, h, w, patch_size)

    labels = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = mask_img[y:y+patch_size, x:x+patch_size]

            isManip = (
                np.sum(patch) > mask_threshold * patch_size**2
            ).astype(np.uint8)
            labels.append(isManip)

    labels = np.array(labels, dtype=np.uint8)

    return labels

def convert2gray(image, binary=False):
    if binary:
        image = image[:, :, 0]
        return image

    if len(image.shape) == 3:
        if image.shape[2] == 4:
            image = image[:, :, :3]
        image = color.rgb2gray(image)
        image = img_as_ubyte(image)
    return image

def resize4patch(image, h, w, patch_size=16):
    if h % patch_size == 0 and w % patch_size == 0:
        return image, h, w

    h = (h // patch_size) * patch_size
    w = (w // patch_size) * patch_size

    image = cv2.resize(
        image,
        (w, h),
        interpolation=cv2.INTER_AREA,
    )
    
    return image, h, w

def patchify(image, h, w, patch_size=32, stride=16):
    patches = []

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patches.append(image[y:y+patch_size, x:x+patch_size])

    print(f'Number of Patches: {len(patches)}')

    return patches

def unpatchify(patches, h, w, patch_size=32, stride=16):
    if len(patches.shape) == 1:
        reconstructed = np.zeros((h, w), dtype=np.float32)
    else:
        channels = patches.shape[-1] if len(patches.shape) > 3 else 1
        shape = (h, w, channels) if channels > 1 else (h, w)
        reconstructed = np.zeros(shape, dtype=np.float32)
        
    overlap_count = np.zeros((h, w), dtype=np.float32)
    
    idx = 0
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            if len(patches.shape) == 1:
                reconstructed[y:y+patch_size, x:x+patch_size] += patches[idx]
            else:
                reconstructed[y:y+patch_size, x:x+patch_size] += patches[idx]
                
            overlap_count[y:y+patch_size, x:x+patch_size] += 1
            idx += 1
            
    overlap_count[overlap_count == 0] = 1
    
    if len(patches.shape) == 1:
        reconstructed = reconstructed / overlap_count
    else:
        reconstructed = reconstructed / overlap_count[:, :, np.newaxis] \
                        if len(reconstructed.shape) == 3 \
                        else reconstructed / overlap_count
        
    return reconstructed

def hannWindow(image, patch_size=16):
    h = np.hanning(patch_size)
    window = np.outer(h, h)

    image_windowed = image * window

    return image_windowed

def computeNoiseRes(image, blur_kernel=5):
    image = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    residual = image - blurred
    # residual = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)

    return residual

def corrcoef(a) -> float:
    x = a[:-1]
    y = a[1:]

    mx = x.mean()
    my = y.mean()

    sx = x.std()
    sy = y.std()

    if sx < 1e-8 or sy < 1e-8:
        corr = 0.0
    else:
        corr = np.mean((x - mx) * (y - my)) / (sx * sy)

    return corr

def skew(a, **kwargs):
    if np.var(a) < 1e-10:
        skewness = 0.0
    else:
        skewness = np_skew(a, **kwargs)

    return skewness

def kurtosis(a, **kwargs):
    if np.var(a) < 1e-10:
        kurt = 0.0
    else:
        kurt = np_kurtosis(a, **kwargs)

    return kurt