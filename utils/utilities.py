import numpy as np
import cv2
from skimage import io, color
from skimage.color import label2rgb
from skimage.util import img_as_ubyte

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
                np.sum(patch) > mask_threshold * patch_size
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

def computeNoiseRes(image, blur_kernel=5):
    blurred = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    residual = image - blurred
    residual = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX)
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

def skew(a) -> float:
    n = len(a)
    m = np.mean(a)
    sd = np.std(a, ddof=1)
    third_moment = np.mean((a - m) ** 3)

    if sd < 1e-8:
        return 0.0
    else:
        skewness = third_moment / (sd ** 3)

    return skewness

def kurtosis(a, fisher=True, bias=True) -> float:
    n = len(a)
    m = np.mean(a)
    sd = np.std(a, ddof=1)
    fourth_moment = np.mean((a - m) ** 4)

    if sd < 1e-8:
        return 0.0
    else:
        kurt = fourth_moment / (sd ** 4)

    if not bias:
        e = a - m
        m2 = np.mean(e ** 2)
        m4 = np.mean(e ** 4)

        k2 = (n / (n - 1)) * m2
        k4 = (n**2 * ((n + 1) * m4 - 3 * (n - 1) * m2**2)) / ((n - 1) * (n - 2) * (n - 3))

        kurt = k4 / (k2 ** 2) if k2 > 1e-8 else 0.0

    kurt -= 3 if fisher else kurt

    return kurt